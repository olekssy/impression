""" Module provides an interface for collaborative filtering models. """

import numpy as np


class UserMemoryModel:
    """ Class for predicting item ratings with the user-based method.

    Methods:
        fit(self, observed_ratings: np.ndarray) -> None:
            Fits model to observed ratings.
        predict(user_id: int, item_id: int) -> float:
            Predicts rating of the target item for a user.
        top_k_items(user_id: int, k: int = 1) -> list[int]:
            Predicts top-k items for a user.
        complete_rating_matrix() -> np.ndarray: 
            Predicts missing ratings for all users.

    Typical usage:
        rating_matrix = np.array([
            [np.nan, 2, 0, np.nan],
            [-2, np.nan, np.nan, 0],
            [1, -1, np.nan, np.nan],
            [1, 0, -1, np.nan],
        ])

        umm = UserMemoryModel()

        # fit model to observed ratings
        umm.fit(rating_matrix)

        # predict rating of item(0) for user(0)
        print(umm.predict(user_id=0, item_id=0))

        # predict top-3 rated items for user(2)
        print(umm.top_k_items(user_id=2, k=3))

        # predict missing ratings for all users
        print(umm.complete_rating_matrix())

        # estimated similarity matrix
        print(umm.sim_scores.round(1))
    """

    def __init__(self) -> None:
        # m x n (immutable) observed rating matrix
        self.observed_ratings: np.ndarray

        # number of users, items
        self.n_users: int
        self.n_items: int

        # m x n (mutable) matrix of predicted and observed ratings
        self.pred_ratings: np.ndarray

        # m x m upper diagonal user similarity score matrix
        self.sim_scores: np.ndarray

        # mean observed user ratings for mean-centering
        self.mu: np.ndarray

        # a flag for fitting a model to observed ratings
        self.is_fit: bool = False

    def _cosine(self, user_1: np.ndarray, user_2: np.ndarray) -> float:
        """ Calculates cosine similarity of two users from mutual ratings.
            Method calculates Pearson correlation similarity score, if passed
            mean-centered rating vectors.

        Args:
            user_1 (np.ndarray): mutualy observed ratings of user/peer.
            user_2 (np.ndarray): mutualy observed ratings of user/peer.

        Returns:
            float: cosine similarity (Pearson correlation) score.
        """

        # edge-case: l2_norm is zero
        l2_prod = np.linalg.norm(user_1) * np.linalg.norm(user_2)
        if round(l2_prod):
            return user_1 @ user_2 / l2_prod
        return 0

    def _get_sim_scores(self, user_id: int) -> np.ndarray:
        """ Gets user-peers similarity scores. Estimates missing score as
            Pearson correlation of mutually observed ratings item ratings.
            Fills missing sim scores in the upper diagonal m x m matrix.
        """

        top = self.sim_scores[:user_id, user_id]
        right = self.sim_scores[user_id, user_id:]
        user_sims = np.concatenate((top, right), axis=None)

        if not np.isnan(user_sims).any():
            return user_sims

        user_ratings = self.observed_ratings[user_id]
        observed_user_map = ~np.isnan(user_ratings)
        missing_sim_peers = np.argwhere(np.isnan(user_sims)).flatten()

        for peer_id in missing_sim_peers:
            peer_ratings = self.observed_ratings[peer_id]
            observed_peer_map = ~np.isnan(peer_ratings)
            observed_mutual_map = observed_user_map & observed_peer_map

            # at least one observed mutual rating required to estimate sim score
            sim_score: float = 0.0
            if user_ratings[observed_mutual_map].size:
                sim_score = self._cosine(
                    user_ratings[observed_mutual_map] - self.mu[user_id],
                    peer_ratings[observed_mutual_map] - self.mu[peer_id])
            self.sim_scores[min(user_id, peer_id),
                            max(user_id, peer_id)] = sim_score

        return np.concatenate((top, right), axis=None)

    def fit(self, observed_ratings: np.ndarray) -> None:
        """ Fits model to observed ratings.
            Compute similarity score of users from observed ratings.
            Sets mean rating for a user.

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

        self.is_fit = True

    def predict(self, user_id: int, item_id: int) -> float:
        """ Predicts rating of the target item for a user.

        Args:
            user_id (int): index of the target user.
            item_id (int): index of the target item.

        Returns:
            float: predicted item rating.
        """

        if not self.is_fit:
            raise Exception('The model must be fit to observed ratings.')

        # get cached rating, if exists
        pred_r: float = self.pred_ratings[user_id, item_id]
        if not np.isnan(pred_r):
            return pred_r

        # get similarity scores and peer ratings
        user_sims = self._get_sim_scores(user_id)
        peer_sims = np.delete(user_sims, user_id)
        peer_sims = np.nan_to_num(peer_sims)

        item_ratings = self.observed_ratings[:, item_id]
        peer_ratings = np.delete(item_ratings, user_id)

        # at least one valid sim-rating pair required to predict rating
        valid_sim = np.where(peer_sims != 0, True, False)
        valid_ratings = ~np.isnan(peer_ratings)
        sr_pair_exists: bool = (valid_sim & valid_ratings).any()

        # similarity norm of peers with observed ratings
        peer_mus = np.delete(self.mu, user_id)
        sim_norm: float = peer_sims @ ~np.isnan(peer_ratings)

        # predict item rating for user as similarity norm weighted dot product
        # of peers similarity and observed peer ratings for the item
        # edge-case: near-zero peer similarity norm (denominator)
        if sr_pair_exists and abs(sim_norm) > 0.2:
            peer_ratings = np.nan_to_num(peer_ratings - peer_mus)
            pred_r = self.mu[user_id] + peer_sims @ peer_ratings / sim_norm
            self.pred_ratings[user_id, item_id] = pred_r

        return pred_r

    def complete_rating_matrix(self) -> np.ndarray:
        """ Predicts missing ratings for all users. """

        for user_id in range(self.n_users):
            for item_id in range(self.n_items):
                self.predict(user_id, item_id)

        return self.pred_ratings

    def top_k_items(self, user_id: int, k: int = 1) -> list[int]:
        """ Predicts top-k items for a user.

        Args:
            user_id (int): target user to make prediction for.
            k (int, optional): number of predicted top rated items.
                Defaults to 1.

        Returns:
            list[int]: indices of predicted top rated items.
        """

        # predict missing ratings for user
        user_ratings: np.ndarray = self.pred_ratings[user_id]
        missing_rating_ids = np.argwhere(np.isnan(user_ratings)).flatten()
        for item_id in missing_rating_ids:
            self.predict(user_id, item_id)

        # select top-k rated items limited by number of predicted
        # and observed ratings
        k = min(k, sum(~np.isnan(user_ratings)))
        user_ratings = np.nan_to_num(user_ratings, nan=-np.inf)

        return np.argsort(-user_ratings)[:k].tolist()
