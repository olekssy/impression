import pytest
import numpy as np
from recommenders.collaborative_filtering import UserMemoryModel


@pytest.fixture
def ratings() -> np.ndarray:
    return np.array([
        [np.nan, 2, 0, np.nan],
        [-2, np.nan, np.nan, 0],
        [1, -1, np.nan, np.nan],
        [1, 0, -1, np.nan],
    ])


@pytest.fixture
def model() -> UserMemoryModel:
    return UserMemoryModel()


def test_hparam_alpha(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.alpha = 2.0
    model.fit(ratings)
    assert model.predict(0, 0) == 2.0
    assert round(model.predict(2, 2), 1) == -0.2


def test_predict_pearson(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.sim_method = 'pearson'
    model.fit(ratings)
    assert model.predict(0, 0) == 2.0
    assert model.predict(2, 2) == -1.0
    assert np.isnan(model.predict(3, 3))


def test_predict_cosine(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.sim_method = 'cosine'
    model.fit(ratings)
    assert np.isnan(model.predict(0, 0))
    assert model.predict(2, 2) == -1.0
    assert np.isnan(model.predict(3, 3))


def test_predict_zscore(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.sim_method = 'z-score'
    model.fit(ratings)
    assert round(model.predict(0, 0), 1) == 2.2
    assert round(model.predict(2, 2), 1) == -1.2
    assert np.isnan(model.predict(3, 3))


def test_mu(model: UserMemoryModel, ratings: np.ndarray) -> None:
    expected = np.array([1., -1., 0., 0.])
    model.fit(ratings)
    np.testing.assert_array_almost_equal(model.mu, expected)


def test_sigma(model: UserMemoryModel, ratings: np.ndarray) -> None:
    expected = np.array([1., 1., 1., 0.8])
    model.fit(ratings)
    np.testing.assert_array_almost_equal(model.sigma, expected, 1)


def test_sim_cosine(model: UserMemoryModel, ratings: np.ndarray) -> None:
    expected = np.array([
        [1., 0., -1., 0.],
        [np.nan, 1., -1., -1.],
        [np.nan, np.nan, 1., 0.7],
        [np.nan, np.nan, np.nan, 1.],
    ])
    model.sim_method = 'cosine'
    model.fit(ratings)
    model.complete_rating_matrix()
    np.testing.assert_array_almost_equal(model.sim_scores, expected, 1)


def test_sim_pearson(model: UserMemoryModel, ratings: np.ndarray) -> None:
    expected = np.array([
        [1., 0., -1., 0.7],
        [np.nan, 1., -1., -1.],
        [np.nan, np.nan, 1., 0.7],
        [np.nan, np.nan, np.nan, 1.],
    ])
    model.sim_method = 'pearson'
    model.fit(ratings)
    model.complete_rating_matrix()
    np.testing.assert_array_almost_equal(model.sim_scores, expected, 1)


def test_sim_zscore(model: UserMemoryModel, ratings: np.ndarray) -> None:
    expected = np.array([
        [1., 0., -1., 0.7],
        [np.nan, 1., -1., -1.],
        [np.nan, np.nan, 1., 0.7],
        [np.nan, np.nan, np.nan, 1.],
    ])
    model.sim_method = 'z-score'
    model.fit(ratings)
    np.testing.assert_array_almost_equal(model.sim_scores, expected, 1)


def test_matrix_completion(model: UserMemoryModel, ratings: np.ndarray) -> None:
    expected = np.array([
        [2., 2., 0., np.nan],
        [-2., np.nan, np.nan, 0.],
        [1., -1., -1., np.nan],
        [1., 0., -1., np.nan],
    ])
    model.fit(ratings)
    np.testing.assert_array_almost_equal(model.complete_rating_matrix(),
                                         expected, 1)


def test_top_k(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.fit(ratings)
    assert model.top_k_items(0, 3) == [0, 1, 2]
    assert model.top_k_items(2, 3) == [0]
