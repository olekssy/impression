""" Module provides an interface for collaborative filtering models. """

import numpy as np


class UserMemoryModel:
    """ Class for predicting item ratings with the user-based method.

    Methods:
        predict(user_id: int, item_id: int) (float): Predicts rating of
            the target item for a user.
        top_k_items(user_id: int, k: int = 1) (list[int]): Predicts top-k items
            for an user.
        complete_rating_matrix() (np.ndarray): Fills missing ratings in the
            rating matrix.

    Typical usage:
        rating_matrix = np.array([
            [np.nan, 2, 0, np.nan],
            [-2, np.nan, np.nan, 0],
            [1, -1, np.nan, np.nan],
            [1, 0, -1, np.nan],
        ])

        umm = UserMemoryModel(rating_matrix)

        # predict rating of item(0) for user(0)
        print(umm.predict(user_id=0, item_id=0))

        # predict top-3 rated items for user(2)
        print(umm.top_k_items(user_id=2, k=3))

        # predict missing ratings for all users
        print(umm.complete_rating_matrix().round(1))

        # estimated similarity matrix
        print(umm.similarity_scores.round(1))
    """

    def __init__(self, rating_matrix: np.ndarray) -> None:
        # m x n (immutable) observed rating matrix
        self.rating_matrix = rating_matrix

        # number of users, items
        self.n_users, self.n_items = self.rating_matrix.shape

        # m x n (mutable) matrix of predicted and observed ratings
        self.cached_ratings: np.ndarray = self.rating_matrix.copy()

        # m x m user similarity score matrix, with diagonal of ones
        self.similarity_scores: np.ndarray = np.full(
            (self.n_users, self.n_users), np.nan)
        np.fill_diagonal(self.similarity_scores, 1.0)

        # mean observed user ratings for mean-centering
        self.mus: np.ndarray = np.nanmean(self.rating_matrix, 1)

    def _cosine(self, user_1: np.ndarray, user_2: np.ndarray) -> float:
        """ Calculates cosine similarity of two users from mutual ratings.
            Method calculates Pearson correlation similarity score, if passed
            mean-centered rating vectors. """

        # edge-case: l2_norm is zero
        l2_prod = np.linalg.norm(user_1) * np.linalg.norm(user_2)
        if round(l2_prod):
            return user_1 @ user_2 / l2_prod
        return 0

    def predict(self, user_id: int, item_id: int) -> float:
        """Predicts rating of the target item for a user.

        Args:
            user_id (int): index of the target user.
            item_id (int): index of the target item.

        Returns:
            float: predicted item rating.
        """

        # get cached rating, if exists
        pred_r: float = self.cached_ratings[user_id, item_id]
        if not np.isnan(pred_r):
            return pred_r

        # predict missing peer similarity scores from mutually observed ratings
        user_ratings = self.rating_matrix[user_id]
        observed_user_map = ~np.isnan(user_ratings)

        user_sims = self.similarity_scores[user_id]
        missing_sim_ids = np.argwhere(np.isnan(user_sims)).flatten()

        for peer_id in missing_sim_ids:
            peer_ratings = self.rating_matrix[peer_id]
            observed_peer_map = ~np.isnan(peer_ratings)
            observed_mutual_map = (observed_user_map & observed_peer_map)

            # at least one observed mutual rating required to estimate sim score
            sim_score: float = 0.0
            if user_ratings[observed_mutual_map].size:
                sim_score = self._cosine(
                    user_ratings[observed_mutual_map] - self.mus[user_id],
                    peer_ratings[observed_mutual_map] - self.mus[peer_id])
            self.similarity_scores[user_id, peer_id] = sim_score

        peer_sims = np.delete(user_sims, user_id)
        peer_sims = np.nan_to_num(peer_sims)

        # avoid near-zero peer similarity norm (denominator)
        sim_norm: float = sum(peer_sims)
        if abs(sim_norm) < 0.2:
            return np.nan

        item_ratings = self.rating_matrix[:, item_id]
        peer_ratings = np.delete(item_ratings, user_id)

        # at least one valid sim-rating pair required to predict rating
        valid_sim = np.where(peer_sims != 0, True, False)
        valid_ratings = ~np.isnan(peer_ratings)
        sr_pair_exists: bool = (valid_sim & valid_ratings).any()

        # predict item rating for user as similarity norm weighted dot product
        # of peers similarity and observed peer ratings for the item
        if sr_pair_exists:
            peer_mus = np.delete(self.mus, user_id)
            peer_ratings = np.nan_to_num(peer_ratings - peer_mus)
            pred_r = self.mus[user_id] + peer_sims @ peer_ratings / sim_norm

        # cache predicted rating
        self.cached_ratings[user_id, item_id] = pred_r

        return pred_r

    def complete_rating_matrix(self) -> np.ndarray:
        """ Fills missing ratings in the rating matrix. """

        for user_id in range(self.n_users):
            for item_id in range(self.n_items):
                self.predict(user_id, item_id)

        return self.cached_ratings

    def top_k_items(self, user_id: int, k: int = 1) -> list[int]:
        """ Predicts top-k items for an user. """

        # predict missing ratings for user
        user_ratings: np.ndarray = self.cached_ratings[user_id]
        missing_rating_ids = np.argwhere(np.isnan(user_ratings)).flatten()
        for item_id in missing_rating_ids:
            self.predict(user_id, item_id)

        # select top-k rated items limited by number of predicted
        # and observed ratings
        k = min(k, sum(~np.isnan(user_ratings)))
        user_ratings = np.nan_to_num(user_ratings, nan=-np.inf)

        return np.argsort(-user_ratings)[:k].tolist()
