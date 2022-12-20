""" This module provides an interface for collaborative filtering models.

    To provide cleaner and more comprehensive abstractions, each model has its
    own implementation w/o inheritance. Keeping classes isolated is a deliberate
    decision, despite the minor code duplication. """

import numpy as np


def cosine_similarity(r_user: np.ndarray,
                      r_peer: np.ndarray,
                      min_denominator: float = 0.1) -> float:
    """ Calculates cosine similarity of two users from mutually observed
        ratings. The method calculates Pearson correlation, if user ratings are
        mean-centered with the user mu. The method calculates adj. cosine
        similarity, if passed user ratings are mean-centered with the item mu.

        Args:
            r_user (np.ndarray): mutualy observed user ratings.
            r_peer (np.ndarray): mutualy observed peer ratings.
            min_denominator (float): min l2 norm to prevent exploding score.

        Returns:
            float: cosine similarity score.
        """

    # edge-case: l2_norm is near-zero
    l2_prod = np.linalg.norm(r_user) * np.linalg.norm(r_peer)
    if abs(l2_prod) > min_denominator:
        return r_user @ r_peer / l2_prod
    return 0.0


class UserMemoryModel:
    """ Class for predicting item ratings with the user-based method.
        It provides an abstraction for solving a matrix completion problem, and
        computing top-k items for a user.

    Hyperparameters:
        alpha (1.0): an exponent that amplifies similarity score magnitude.
        min_sim (0.1): select peers with positive significant similarity scores.
        sim_method (cosine, pearson, z-score): method for predicting ratings
        and similarity scores.

    Methods:
        fit(observed_ratings: np.ndarray) -> None:
            Fits model to observed ratings. Computes similarity score matrix.
        similarity(user_id: int) -> np.ndarray:
            Gets user-peers similarity scores.
        predict(user_id: int, item_id: int) -> float:
            Predicts rating of the target item for a user.
        top_k_items(user_id: int,
                    k: int = 1,
                    min_rating: float = 0) -> list[int]:
            Predicts top-k items for a user.
        complete_rating_matrix() -> np.ndarray:
            Predicts missing ratings for all users.
    """

    def __init__(self,
                 sim_method: str = 'pearson',
                 alpha: float = 1.0,
                 min_similarity: float = 0.1) -> None:
        # m x n (immutable) observed rating matrix
        self.observed_ratings: np.ndarray

        # number of users, items
        self.n_users: int
        self.n_items: int

        # m x n (mutable) matrix of predicted and observed ratings
        self.pred_ratings: np.ndarray

        # m x m upper diagonal user similarity score matrix
        self.similarity_scores: np.ndarray

        # mean, std of observed user ratings for mean-centering and z-scores
        self.mu: np.ndarray
        self.sigma: np.ndarray

        # == Hyperparameters ==

        # prevents exploding sim scores and pred ratings
        self.min_denominator: float = 0.1

        # similarity score methods
        self.valid_pred_methods: list[str] = ['cosine', 'pearson', 'z-score']
        if sim_method not in self.valid_pred_methods:
            raise ValueError(
                'Invalid similarity method. Select from the following:',
                self.valid_pred_methods)
        self.sim_method: str = sim_method

        # minimum significant similarity score to include peer ratings
        # negative or near-zero sims do contribute to rating prediction
        self.min_sim = min_similarity

        # h-parameter that amplifies the sim scores magnitude
        self.alpha = alpha

    def fit(self, observed_ratings: np.ndarray) -> None:
        """ Fits model attrs to observed ratings. Sets mean rating for a user.
            Computes similarity score matrix.

        Args:
            observed_ratings (np.ndarray): observed user-item ratings matrix.
            sim_method (str): method for estimating similarity score. Available
                methods are listed in the valid_sim_methods attr.
            min_sim (float): min similarity required to include peer ratings
                into prediction.
        """

        # init model attributes
        self.observed_ratings = observed_ratings
        self.n_users, self.n_items = self.observed_ratings.shape
        self.pred_ratings = self.observed_ratings.copy()

        self.similarity_scores = np.full((self.n_users, self.n_users), np.nan)
        np.fill_diagonal(self.similarity_scores, 1.0)

        self.mu = np.nanmean(self.observed_ratings, 1)
        self.sigma = np.nanstd(self.observed_ratings, 1)
        # edge-case: set near-zero sigma (denominator) to one
        self.sigma = np.where(
            abs(self.sigma) < self.min_denominator, 1, self.sigma)

        # compute similarity matrix
        for user_id in range(self.n_users):
            self.similarity(user_id)

    def similarity(self, user_id: int) -> np.ndarray:
        """ Gets user-peers similarity scores. Estimates missing sim score of
            mutually observed item ratings.
            Fills missing sim scores in the upper diagonal m x m matrix.
        """

        top = self.similarity_scores[:user_id, user_id]
        right = self.similarity_scores[user_id, user_id:]
        user_sims = np.concatenate((top, right), axis=None)
        missing_sims_map = np.isnan(user_sims)

        if not missing_sims_map.any():
            return user_sims

        user_ratings = self.observed_ratings[user_id]
        user_map = ~np.isnan(user_ratings)
        missing_sims = np.argwhere(missing_sims_map).flatten()

        for peer_id in missing_sims:
            peer_ratings = self.observed_ratings[peer_id]
            peer_map = ~np.isnan(peer_ratings)
            mutual_map = user_map & peer_map

            # at least one observed mutual rating required to estimate sim score
            sim_score: float = 0.0
            if user_ratings[mutual_map].size:
                # base case cosine similarity
                r_user = user_ratings[mutual_map]
                r_peer = peer_ratings[mutual_map]

                if self.sim_method == 'pearson':
                    r_user -= self.mu[user_id]
                    r_peer -= self.mu[peer_id]

                elif self.sim_method == 'z-score':
                    r_user = (r_user - self.mu[user_id]) / self.sigma[user_id]
                    r_peer = (r_peer - self.mu[peer_id]) / self.sigma[peer_id]

                sim_score = cosine_similarity(r_user, r_peer,
                                              self.min_denominator)

            self.similarity_scores[min(user_id, peer_id),
                                   max(user_id, peer_id)] = sim_score

        return np.concatenate((top, right), axis=None)

    def predict(self, user_id: int, item_id: int) -> float:
        """ Predicts rating of the target item for a user.

        Args:
            user_id (int): index of the target user.
            item_id (int): index of the target item.

        Returns:
            float: predicted item rating.
        """

        # get cached rating, if exists
        pred_r: float = self.pred_ratings[user_id, item_id]
        if not np.isnan(pred_r):
            return pred_r

        # select significant similarity scores and peer ratings for a user
        user_sims = self.similarity(user_id)
        peer_sims = np.delete(user_sims, user_id)
        peer_sims = np.nan_to_num(peer_sims)
        peer_sims = np.power(peer_sims, self.alpha)
        peer_sims *= np.where(peer_sims >= self.min_sim, peer_sims, 0)

        item_ratings = self.observed_ratings[:, item_id]
        peer_ratings = np.delete(item_ratings, user_id)

        # at least one valid sim-rating pair required to predict rating
        valid_sim = np.where(peer_sims != 0, True, False)
        valid_ratings = ~np.isnan(peer_ratings)
        peers_exist: bool = (valid_sim & valid_ratings).any()

        # similarity mu, sigma, norm of peers with observed ratings
        peer_mus = np.delete(self.mu, user_id)
        peer_sigma = np.delete(self.sigma, user_id)
        sim_norm: float = peer_sims @ valid_ratings

        # predict item rating for a user as similarity norm weighted dot product
        # of peers similarity and observed peer ratings for the item
        # edge-case: near-zero peer similarity norm (denominator)
        if not peers_exist and abs(sim_norm) < self.min_denominator:
            return pred_r

        peer_ratings = np.nan_to_num(peer_ratings)
        if self.sim_method == 'cosine':
            pred_r = peer_sims @ peer_ratings / sim_norm

        elif self.sim_method == 'pearson':
            peer_ratings -= peer_mus
            pred_r = self.mu[user_id] + peer_sims @ peer_ratings / sim_norm

        elif self.sim_method == 'z-score':
            peer_ratings = (peer_ratings - peer_mus) / peer_sigma
            pred_r = (self.mu[user_id] +
                      self.sigma[user_id] * peer_sims @ peer_ratings / sim_norm)

        self.pred_ratings[user_id, item_id] = pred_r

        return pred_r

    def complete_rating_matrix(self) -> np.ndarray:
        """ Predicts missing ratings for all users. """

        for user_id in range(self.n_users):
            for item_id in range(self.n_items):
                self.predict(user_id, item_id)

        return self.pred_ratings

    def top_k_items(self,
                    user_id: int,
                    k: int = 1,
                    min_rating: float = 0) -> list[int]:
        """ Predicts top-k items for a user.

        Args:
            user_id (int): target user to make prediction for.
            k (int, optional): number of predicted top rated items.
                Defaults to 1.
            min_rating (int, optional): min positive rating to be considered in
                recommendation.

        Returns:
            list[int]: indices of predicted top rated items.
        """

        # predict missing ratings for user
        user_ratings: np.ndarray = self.pred_ratings[user_id]
        missing_rating_ids = np.argwhere(np.isnan(user_ratings)).flatten()
        for item_id in missing_rating_ids:
            self.predict(user_id, item_id)

        # select top-k rated items limited by the number of positive ratings
        user_ratings = np.where(user_ratings < min_rating, np.nan, user_ratings)
        k = min(k, sum(~np.isnan(user_ratings)))
        user_ratings = np.nan_to_num(user_ratings, nan=-np.inf)

        return np.argsort(-user_ratings)[:k].tolist()


class ItemMemoryModel:
    """ Class for predicting item ratings with the item-based method.
        It provides an abstraction for solving a matrix completion problem, and
        computing top-k items for a user.

    Hyperparameters:
        alpha (1.0): an exponent that amplifies similarity score magnitude.
        min_sim (0.1): select peers with positive significant similarity scores.

    Methods:
        fit(observed_ratings: np.ndarray) -> None:
            Fits model to observed ratings. Computes similarity score matrix.
        similarity(user_id: int) -> np.ndarray:
            Gets item-comps adj. cosine similarity scores.
        predict(user_id: int, item_id: int) -> float:
            Predicts rating of the target item for a user.
        top_k_items(user_id: int,
                    k: int = 1,
                    min_rating: float = 0) -> list[int]:
            Predicts top-k items for a user.
        complete_rating_matrix() -> np.ndarray:
            Predicts missing ratings for all users.
    """

    def __init__(self, alpha: float = 1.0, min_similarity: float = 0.1) -> None:
        # m x n (immutable) observed rating matrix
        self.observed_ratings: np.ndarray

        # number of users, items
        self.n_users: int
        self.n_items: int

        # m x n (mutable) matrix of predicted and observed ratings
        self.pred_ratings: np.ndarray

        # m x m upper diagonal user similarity score matrix
        self.similarity_scores: np.ndarray

        # mean, std of observed user ratings for mean-centering and z-scores
        self.mu: np.ndarray
        self.sigma: np.ndarray

        # == Hyperparameters ==

        # prevents exploding sim scores and pred ratings
        self.min_denominator: float = 0.1

        # minimum significant similarity score to include peer ratings
        # negative or near-zero sims do contribute to rating prediction
        self.min_sim = min_similarity

        # h-parameter that amplifies the sim scores magnitude
        self.alpha = alpha

        # mean-centered observed rating matrix
        self.mc_ratings: np.ndarray

    def fit(self, observed_ratings: np.ndarray) -> None:
        """ Fits model attrs to observed ratings.
            Computes similarity score matrix.

        Args:
            observed_ratings (np.ndarray): observed user-item ratings matrix.
            sim_method (str): method for estimating similarity score. Available
                methods are listed in the valid_sim_methods attr.
            min_sim (float): min similarity required to include peer ratings
                into prediction.
        """

        # init model attributes
        self.observed_ratings = observed_ratings
        self.n_users, self.n_items = self.observed_ratings.shape
        self.pred_ratings = self.observed_ratings.copy()

        self.mu = np.nanmean(self.observed_ratings, 1)
        self.sigma = np.nanstd(self.observed_ratings, 1)
        # edge-case: set near-zero sigma (denominator) to one
        self.sigma = np.where(
            abs(self.sigma) < self.min_denominator, 1, self.sigma)

        self.similarity_scores = np.full((self.n_items, self.n_items), np.nan)
        np.fill_diagonal(self.similarity_scores, 1.0)

        # mean-center user ratings
        self.mc_ratings = (self.observed_ratings -
                           self.mu.reshape(self.n_users, 1))

        # compute similarity score matrix
        for item_id in range(self.n_items):
            self.similarity(item_id)

    def similarity(self, item_id: int) -> np.ndarray:
        """ Gets item-comps adj. cosine similarity scores. Estimates missing
            sim score of mutually observed ratings. Fills missing sim scores
            in the upper diagonal n x n matrix.
        """

        top = self.similarity_scores[:item_id, item_id]
        right = self.similarity_scores[item_id, item_id:]
        item_sims = np.concatenate((top, right), axis=None)
        missing_sims_map = np.isnan(item_sims)

        if not missing_sims_map.any():
            return item_sims

        item_ratings = self.mc_ratings[:, item_id]
        item_map = ~np.isnan(item_ratings)
        missing_sims = np.argwhere(missing_sims_map).flatten()

        for comp_id in missing_sims:
            comp_ratings = self.mc_ratings[:, comp_id]
            comp_map = ~np.isnan(comp_ratings)
            mutual_map = item_map & comp_map

            # at least one observed mutual rating required to estimate sim score
            sim_score: float = 0.0
            if item_ratings[mutual_map].size:
                r_item = item_ratings[mutual_map]
                r_comp = comp_ratings[mutual_map]

                sim_score = cosine_similarity(r_item, r_comp,
                                              self.min_denominator)

            self.similarity_scores[min(item_id, comp_id),
                                   max(item_id, comp_id)] = sim_score

        return np.concatenate((top, right), axis=None)

    def predict(self, user_id: int, item_id: int) -> float:
        """ Predicts rating of the target item for a user.

        Args:
            user_id (int): index of the target user.
            item_id (int): index of the target item.

        Returns:
            float: predicted item rating.
        """

        # get cached rating, if exists
        pred_r: float = self.pred_ratings[user_id, item_id]
        if not np.isnan(pred_r):
            return pred_r

        # select significant similarity scores and peer ratings for a user
        item_sims = self.similarity(item_id)
        comp_sims = np.delete(item_sims, item_id)
        comp_sims = np.nan_to_num(comp_sims)
        comp_sims = np.power(comp_sims, self.alpha)
        comp_sims *= np.where(comp_sims >= self.min_sim, comp_sims, 0.0)

        user_ratings = self.observed_ratings[user_id]
        comp_ratings = np.delete(user_ratings, item_id)

        # at least one valid sim-rating pair required to predict rating
        valid_sim = np.where(comp_sims != 0, True, False)
        valid_ratings = ~np.isnan(comp_ratings)
        peers_exist: bool = (valid_sim & valid_ratings).any()

        # similarity norm of the others user items with observed ratings
        sim_norm: float = comp_sims @ valid_ratings

        # predict item rating for a user as similarity norm weighted dot product
        # of peers similarity and observed peer ratings for the item
        # edge-case: near-zero peer similarity norm (denominator)
        if not peers_exist and abs(sim_norm) < self.min_denominator:
            return pred_r

        comp_ratings = np.nan_to_num(comp_ratings)
        pred_r = comp_sims @ comp_ratings / sim_norm

        self.pred_ratings[user_id, item_id] = pred_r

        return pred_r

    def complete_rating_matrix(self) -> np.ndarray:
        """ Predicts missing ratings for all users. """

        for user_id in range(self.n_users):
            for item_id in range(self.n_items):
                self.predict(user_id, item_id)

        return self.pred_ratings

    def top_k_items(self,
                    user_id: int,
                    k: int = 1,
                    min_rating: float = 0) -> list[int]:
        """ Predicts top-k items for a user.

        Args:
            user_id (int): target user to make prediction for.
            k (int, optional): number of predicted top rated items.
                Defaults to 1.
            min_rating (int, optional): min positive rating to be considered in
                recommendation.

        Returns:
            list[int]: indices of predicted top rated items.
        """

        # predict missing ratings for user
        user_ratings: np.ndarray = self.pred_ratings[user_id]
        missing_rating_ids = np.argwhere(np.isnan(user_ratings)).flatten()
        for item_id in missing_rating_ids:
            self.predict(user_id, item_id)

        # select top-k rated items limited by the number of positive ratings
        user_ratings = np.where(user_ratings < min_rating, np.nan, user_ratings)
        k = min(k, sum(~np.isnan(user_ratings)))
        user_ratings = np.nan_to_num(user_ratings, nan=-np.inf)

        return np.argsort(-user_ratings)[:k].tolist()
