import pytest
import numpy as np
from recommenders.collaborative_filtering import UserMemoryModel


@pytest.fixture
def ratings() -> np.ndarray:
    return np.array([
        [np.nan, 2, 0, np.nan, 1, -1],
        [-2, np.nan, np.nan, 0, np.nan, 1],
        [1, -1, np.nan, np.nan, 0, np.nan],
        [1, 0, -1, np.nan, 2, -2],
    ])


@pytest.fixture
def model() -> UserMemoryModel:
    return UserMemoryModel()


def test_svd(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.factorization_method = 'svd'
    model.fit(ratings)
    expected = np.array([
        [1., -0.12, -0.43, 0.03],
        [np.nan, 1., -0.61, -0.79],
        [np.nan, np.nan, 1., 0.16],
        [np.nan, np.nan, np.nan, 1.],
    ])
    np.testing.assert_array_almost_equal(model.sim_scores, expected, 2)


def test_truncated_svd(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.factorization_method = 'svd'
    model.approximate_factorization = True
    model.fit(ratings)
    expected = np.array([
        [1., -0.48, -0.53, 0.14],
        [np.nan, 1., -0.17, -0.9],
        [np.nan, np.nan, 1., 0.18],
        [np.nan, np.nan, np.nan, 1.],
    ])
    np.testing.assert_array_almost_equal(model.sim_scores, expected, 2)


def test_pca(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.factorization_method = 'pca'
    model.fit(ratings)
    expected = np.array([
        [1., -0.06, -0.51, -0.07],
        [np.nan, 1., -0.55, -0.78],
        [np.nan, np.nan, 1., 0.11],
        [np.nan, np.nan, np.nan, 1.],
    ])
    np.testing.assert_array_almost_equal(model.sim_scores, expected, 2)


def test_truncated_pca(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.factorization_method = 'pca'
    model.approximate_factorization = True
    model.fit(ratings)
    expected = np.array([
        [1., -0.41, -0.65, 0.07],
        [np.nan, 1., -0.09, -0.87],
        [np.nan, np.nan, 1., 0.08],
        [np.nan, np.nan, np.nan, 1.],
    ])
    np.testing.assert_array_almost_equal(model.sim_scores, expected, 2)


def test_compressed_pca(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.factorization_method = 'pca'
    model.compression_rate = 0.5
    model.fit(ratings)
    expected = np.array([
        [1., 0.88, -0.94, -0.45],
        [np.nan, 1., -0.66, -0.82],
        [np.nan, np.nan, 1., 0.11],
        [np.nan, np.nan, np.nan, 1.],
    ])
    np.testing.assert_array_almost_equal(model.sim_scores, expected, 2)
