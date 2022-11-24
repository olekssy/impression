""" Module provides an interface for collaborative filtering models. """

import numpy as np


class Similarity:
    """ Class provides functionals for estimating similarity score. """

    @classmethod
    def cosine(cls, user_1: np.ndarray, user_2: np.ndarray) -> float:
        """ Calculates cosine similarity of two users from mutual ratings. """

        # edge-case: l2_norm is zero
        l2_prod = np.linalg.norm(user_1) * np.linalg.norm(user_2)
        if round(l2_prod):
            return user_1 @ user_2 / l2_prod
        return 0.0

    @classmethod
    def pearson(cls, user_1: np.ndarray, user_2: np.ndarray) -> float:
        """ Calculates Pearson correlation of two users from mutual ratings. """

        return cls.cosine(user_1 - np.mean(user_1), user_2 - np.mean(user_2))


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

        # predict rating of item(2) for user(2)
        rating: float = umm.predict(user_id=2, item_id=2)

        # predict missing ratings for all users
        pred_ratings: np.ndarray = umm.complete_rating_matrix()

        # predict top-3 rated items for user(2)
        top_items: list[int] = umm.top_k_items(user_id=2, k=3)
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
        self.rating_mus: np.ndarray = np.nanmean(self.rating_matrix, 1)

    def predict(self, user_id: int, item_id: int) -> float:
        """ Predicts rating of the target item for a user. """

        # get cached rating, if exists
        user_item_rating: float = self.cached_ratings[user_id, item_id]
        if not np.isnan(user_item_rating):
            return user_item_rating

        # predict missing peer similarity scores from observed mutual ratings
        user_ratings: np.ndarray = self.rating_matrix[user_id]
        observed_user_ratings: np.ndarray = ~np.isnan(user_ratings)
        missing_sim_ids: np.ndarray = np.argwhere(
            np.isnan(self.similarity_scores[user_id])).flatten()

        for peer_id in missing_sim_ids:
            peer_ratings: np.ndarray = self.rating_matrix[peer_id]
            observed_peer_ratings: np.ndarray = ~np.isnan(peer_ratings)
            observed_mutual_ratings: np.ndarray = (observed_user_ratings &
                                                   observed_peer_ratings)

            # at least one observed mutual rating required to estimate sim score
            sim_score: float = 0.0
            if user_ratings[observed_mutual_ratings].size:
                sim_score = Similarity.pearson(
                    user_ratings[observed_mutual_ratings],
                    peer_ratings[observed_mutual_ratings])
            self.similarity_scores[user_id, peer_id] = round(sim_score, 2)

        # predict item rating from peers as a sum of observed peer ratings for
        # the target item weighted by similarity score, if combination of peer
        # similarity and peer rating exists
        peer_sim = np.nan_to_num(self.similarity_scores[user_id])
        peer_sim = np.delete(peer_sim, user_id)
        peer_sim_sum: float = sum(peer_sim)

        # similar peers do not exist
        if not round(peer_sim_sum):
            return np.nan

        peer_ratings = self.rating_matrix[:, item_id]
        peer_ratings = np.delete(peer_ratings, user_id)

        # at least one valid similarity-rating pair required for prediction
        similarity_exists = np.where(peer_sim != 0, True, False)
        rating_exists = ~np.isnan(peer_ratings)
        sr_pair_exists: bool = (similarity_exists & rating_exists).any()

        if sr_pair_exists:
            peer_mus = np.delete(self.rating_mus, user_id)
            peer_ratings = np.nan_to_num(peer_ratings - peer_mus)
            user_item_rating = (self.rating_mus[user_id] +
                                peer_sim @ peer_ratings / peer_sim_sum)

        # cache predicted rating
        self.cached_ratings[user_id, item_id] = user_item_rating

        return user_item_rating

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

        # select top-k rated items limited by number of predicted,
        # and observed ratings
        k = min(k, sum(~np.isnan(user_ratings)))
        user_ratings = np.nan_to_num(self.cached_ratings[user_id], nan=-np.inf)

        return np.argsort(-user_ratings)[:k].tolist()
