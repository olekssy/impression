""" This module provides an interface for collaborative filtering models. """

import numpy as np


class GenericMemoryModel:
    """ A generic neighborhood model class provides attribute declarations and
        shared methods for neighborhood based collaborative filtering models.
    """

    def __init__(self) -> None:

        # == Collaborative filtering attributes ==

        # m x n (immutable) observed rating matrix
        self.observed_ratings: np.ndarray

        # number of users, items
        self.n_users: int
        self.n_items: int

        # m x n (mutable) matrix of predicted and observed ratings
        self.pred_ratings: np.ndarray

        # upper diagonal similarity score matrix
        # m x m for user-based CF, n x n for item-based CF
        self.sim_scores: np.ndarray

        # mean, std of observed user (item) ratings for prediction
        self.mu: np.ndarray
        self.sigma: np.ndarray

        # == Hyperparameters ==

        # prevents exploding sim scores and pred ratings
        self.min_denominator: float = 0.1

        # similarity score methods
        self.valid_sim_methods: list[str]
        self.sim_method: str

        # minimum significant similarity score of a peer rating for prediction
        # negative or near-zero sims do contribute to rating prediction
        self.min_sim: float

        # an alpha ≥ 1.0 exponent amplifies the weight of high sim scores
        self.alpha: float

        # dimensionality reduction methods for observed ratings
        self.valid_compression_methods: list[str]
        self.compression_method: str

        # the compression rate controls the rating matrix dimensionality
        # higher compression rate speed up fitting the sim score matrix
        self.compression_rate: float

    def cosine_similarity(self, r_target: np.ndarray,
                          r_peer: np.ndarray) -> float:
        """ Calculates cosine similarity of mutually observed ratings.
            The method calculates Pearson correlation of mean-centered ratings.
            The method calculates adj. cosine similarity of user ratings
            mean-centered with the item mu.

        Args:
            r_target (np.ndarray): mutualy observed user ratings.
            r_peer (np.ndarray): mutualy observed peer ratings.

        Returns:
            float: cosine similarity score, 0.0 for near-zero l2 norm.
        """

        # edge-case: l2_norm is near-zero
        l2_prod = np.linalg.norm(r_target) * np.linalg.norm(r_peer)
        if abs(l2_prod) > self.min_denominator:
            return r_target @ r_peer / l2_prod
        return 0.0


class UserMemoryModel(GenericMemoryModel):
    """ Class for predicting item ratings for the target user with the
        user-based neighborhood method. It provides an abstraction for solving
        a matrix completion problem and predicting the top-k items for a user.

        The class supports a dimensionality reduction of the observed rating
        matrix with SVD, PCA compression methods for speeding up fitting
        similarity matrix. Both compression methods provide more robust
        predictions with mean-centering the rating matrix across users' ratings
        (rows) and items' ratings (cols).

    Hyperparameters:
        similarity_method (cosine, pearson, z-score): a method for fitting
            similarity scores and predicting target ratings.
        min_similarity (≥0.1): minimum significant similarity score of a peer
            rating for prediction.
        alpha (≥1.0): an exponent amplifies the weight of high sim scores.
        compression_method (none, svd, pca): a dimensionality reduction
            method for observed ratings.
        compression_rate (0 ≤ r < 1): controls the rating matrix dimensionality.

    Methods:
        fit(observed_ratings: np.ndarray) -> None:
            Sets model attributes, fits similarity score matrix to ratings.
        predict(user_id: int, item_id: int) -> float:
            Predicts rating of the target item for a user.
        complete_rating_matrix() -> np.ndarray:
            Predicts missing ratings for all users.
        top_k_items(user_id: int,
                    k: int = 1,
                    min_rating: float = 0) -> list[int]:
            Predicts top-k items for a user.
    """

    def __init__(self,
                 similarity_method: str = 'pearson',
                 min_similarity: float = 0.1,
                 alpha: float = 1.0,
                 compression_method: str = 'none',
                 compression_rate: float = 0.0) -> None:

        # shared model attributes
        super().__init__()

        # == Hyperparameters ==

        self.min_sim = max(min_similarity, 0.1)
        self.alpha = max(alpha, 1.0)
        self.compression_rate = max(compression_rate, 0.0)

        # similarity score method
        self.valid_sim_methods = ['cosine', 'pearson', 'z-score']
        if similarity_method not in self.valid_sim_methods:
            raise ValueError(
                'Passed invalid sim method %s. Select from the following:',
                similarity_method, self.valid_sim_methods)
        self.sim_method = similarity_method

        # rating matrix dimensionality reduction method
        self.valid_compression_methods = ['none', 'svd', 'pca']
        if compression_method not in self.valid_compression_methods:
            raise ValueError(
                'Invalid compression method. Select from the following:',
                self.valid_compression_methods)
        self.compression_method: str = compression_method

    def fit(self, observed_ratings: np.ndarray) -> None:
        """ Sets model attributes from observed ratings.
            Fits similarity score matrix to ratings.

        Args:
            observed_ratings (np.ndarray): observed user-item ratings matrix.
        """

        # init model attributes
        self.observed_ratings = observed_ratings
        self.n_users, self.n_items = self.observed_ratings.shape
        self.pred_ratings = self.observed_ratings.copy()

        self.sim_scores = np.full((self.n_users, self.n_users), np.nan)
        np.fill_diagonal(self.sim_scores, 1.0)

        self.mu = np.nanmean(self.observed_ratings, 1)
        self.sigma = np.nanstd(self.observed_ratings, 1)
        # edge-case: set near-zero sigma (denominator) to one
        self.sigma = np.where(self.sigma < self.min_denominator, 1, self.sigma)

        comp_ratings = self.observed_ratings.copy()
        if self.compression_method in ['svd', 'pca']:
            # mean-center observed ratings along rows, cols
            mu = np.nanmean(self.observed_ratings, 1)
            comp_ratings = (self.observed_ratings - mu.reshape(self.n_users, 1))
            comp_ratings -= np.nanmean(comp_ratings, 0)

            # fill missing ratings with users mean
            row_mean = np.nanmean(comp_ratings, 1).reshape(self.n_users, 1)
            comp_ratings = np.where(np.isnan(comp_ratings), row_mean,
                                    comp_ratings)

            # mean-centered full rating matrix along cols
            if self.compression_method == 'pca':
                comp_ratings -= np.mean(comp_ratings, 0)

            # estimate (covariance) matrix n x n
            # get full n x n eigenvector matrix sorted by eigvals decs
            vcm = comp_ratings.T @ comp_ratings / (self.n_items - 1)
            eigvecs = np.linalg.svd(vcm)[0]  # returns full (u, d, u_T)

            # compressed m x d ratings representation with d-highest eigvals
            # the num of top-n features is capped by n_items
            n_dim = int(self.n_items * (1.0 - self.compression_rate))
            comp_ratings = comp_ratings @ eigvecs[:, :max(1, n_dim)]

        # fit similarity matrix
        comp_mu = np.nanmean(comp_ratings, 1)
        comp_sigma = np.nanstd(comp_ratings, 1)
        # edge-case: set near-zero sigma (denominator) to one
        comp_sigma = np.where(comp_sigma < self.min_denominator, 1, comp_sigma)

        for user_id in range(self.n_users):
            self.similarity(user_id, comp_ratings, comp_mu, comp_sigma)

    def similarity(self, user_id: int, rating_matrix: np.ndarray,
                   mu: np.ndarray, sigma: np.ndarray) -> None:
        """ Estimates missing sim score of mutually observed item ratings.
            Fills missing sim scores in the upper diagonal m x m matrix. """

        top = self.sim_scores[:user_id, user_id]
        right = self.sim_scores[user_id, user_id:]
        user_sims = np.concatenate((top, right), axis=None)
        missing_sims_map = np.isnan(user_sims)

        user_ratings = rating_matrix[user_id]
        user_map = ~np.isnan(user_ratings)
        missing_sims = np.argwhere(missing_sims_map).flatten()

        for peer_id in missing_sims:
            peer_ratings = rating_matrix[peer_id]
            peer_map = ~np.isnan(peer_ratings)
            mutual_map = user_map & peer_map

            # at least one observed mutual rating required to estimate sim score
            sim_score: float = 0.0
            if user_ratings[mutual_map].size:
                # base case cosine similarity
                r_user = user_ratings[mutual_map]
                r_peer = peer_ratings[mutual_map]

                if self.sim_method == 'pearson':
                    r_user -= mu[user_id]
                    r_peer -= mu[peer_id]

                elif self.sim_method == 'z-score':
                    r_user = (r_user - mu[user_id]) / sigma[user_id]
                    r_peer = (r_peer - mu[peer_id]) / sigma[peer_id]

                sim_score = super().cosine_similarity(r_user, r_peer)

            self.sim_scores[min(user_id, peer_id),
                            max(user_id, peer_id)] = sim_score

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
        top = self.sim_scores[:user_id, user_id]
        right = self.sim_scores[user_id, user_id:]
        user_sims = np.concatenate((top, right), axis=None)
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

        # predict item rating for a user with sim score and observed peer rating
        # edge-case: near-zero peer sim norm (denominator)
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


class ItemMemoryModel(GenericMemoryModel):
    """ Class for predicting item ratings for the target user with the
        item-based neighborhood method. It provides an abstraction for solving
        a matrix completion problem and predicting the top-k items for a user.

        The class predicts the rating of an item from other similar items
        rated by the target user. It provides more tailored recommendations for
        a user, compared to the user-based neighborhood.

    Hyperparameters:
        min_similarity (≥0.1): minimum significant similarity score of a peer
            rating for prediction.
        alpha (≥1.0): an exponent amplifies the weight of high sim scores.

    Methods:
        fit(observed_ratings: np.ndarray) -> None:
            Sets model attributes, fits similarity score matrix to ratings.
        predict(user_id: int, item_id: int) -> float:
            Predicts rating of the target item for a user.
        complete_rating_matrix() -> np.ndarray:
            Predicts missing ratings for all users.
        top_k_items(user_id: int,
                    k: int = 1,
                    min_rating: float = 0) -> list[int]:
            Predicts top-k items for a user.
    """

    def __init__(self, alpha: float = 1.0, min_similarity: float = 0.1) -> None:

        # shared model attributes
        super().__init__()

        # == Hyperparameters ==

        self.min_sim = max(min_similarity, 0.1)
        self.alpha = max(alpha, 1.0)

    def fit(self, observed_ratings: np.ndarray) -> None:
        """ Sets model attributes from observed ratings.
            Fits similarity score matrix to ratings.

        Args:
            observed_ratings (np.ndarray): observed user-item ratings matrix.
        """

        # init model attributes
        self.observed_ratings = observed_ratings
        self.n_users, self.n_items = self.observed_ratings.shape
        self.pred_ratings = self.observed_ratings.copy()

        self.mu = np.nanmean(self.observed_ratings, 1)

        self.sim_scores = np.full((self.n_items, self.n_items), np.nan)
        np.fill_diagonal(self.sim_scores, 1.0)

        # mean-center observed ratings to remove user sentiment bias
        rating_matrix = self.observed_ratings - self.mu.reshape(self.n_users, 1)

        # fit similarity score matrix
        for item_id in range(self.n_items):
            self.similarity(item_id, rating_matrix)

    def similarity(self, item_id: int, rating_matrix: np.ndarray) -> np.ndarray:
        """ Fits item-peers similarity to mutually observed ratings with
            adj. with cosine similarity scores. Fills missing sim scores in
            the upper diagonal n x n matrix.
        """

        top = self.sim_scores[:item_id, item_id]
        right = self.sim_scores[item_id, item_id:]
        item_sims = np.concatenate((top, right), axis=None)
        missing_sims_map = np.isnan(item_sims)

        item_ratings = rating_matrix[:, item_id]
        item_map = ~np.isnan(item_ratings)
        missing_sims = np.argwhere(missing_sims_map).flatten()

        for comp_id in missing_sims:
            comp_ratings = rating_matrix[:, comp_id]
            comp_map = ~np.isnan(comp_ratings)
            mutual_map = item_map & comp_map

            # at least one observed mutual rating required to estimate sim score
            sim_score: float = 0.0
            if item_ratings[mutual_map].size:
                r_item = item_ratings[mutual_map]
                r_comp = comp_ratings[mutual_map]

                sim_score = super().cosine_similarity(r_item, r_comp)

            self.sim_scores[min(item_id, comp_id),
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
        top = self.sim_scores[:item_id, item_id]
        right = self.sim_scores[item_id, item_id:]
        item_sims = np.concatenate((top, right), axis=None)
        peer_sims = np.delete(item_sims, item_id)
        peer_sims = np.nan_to_num(peer_sims)
        peer_sims = np.power(peer_sims, self.alpha)
        peer_sims *= np.where(peer_sims >= self.min_sim, peer_sims, 0.0)

        user_ratings = self.observed_ratings[user_id]
        peer_ratings = np.delete(user_ratings, item_id)

        # at least one valid sim-rating pair required to predict rating
        valid_sim = np.where(peer_sims != 0, True, False)
        valid_ratings = ~np.isnan(peer_ratings)
        peers_exist: bool = (valid_sim & valid_ratings).any()

        # similarity norm of the others user items with observed ratings
        sim_norm: float = peer_sims @ valid_ratings

        # predict item rating for a user as similarity norm weighted dot product
        # of peers similarity and observed peer ratings for the item
        # edge-case: near-zero peer similarity norm (denominator)
        if not peers_exist and abs(sim_norm) < self.min_denominator:
            return pred_r

        peer_ratings = np.nan_to_num(peer_ratings)
        pred_r = peer_sims @ peer_ratings / sim_norm

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
