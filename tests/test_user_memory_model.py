import pytest
import numpy as np
from recommenders.collaborative import UserMemoryModel


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


def test_predict(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.fit(ratings)
    assert model.predict(0, 0) == 2.0
    assert model.predict(2, 2) == -1.0
    assert model.predict(3, 3) == 1.0


def test_matrix_completion(model: UserMemoryModel, ratings: np.ndarray) -> None:
    expected = np.array([
        [2., 2., 0., np.nan],
        [-2., -1.5, -2., 0.],
        [1., -1., -1., 1.],
        [1., 0., -1., 1.],
    ])
    model.fit(ratings)
    np.testing.assert_array_almost_equal(model.complete_rating_matrix(),
                                         expected)


def test_sim_scores(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.fit(ratings)
    np.testing.assert_array_almost_equal(model._get_sim_scores(0),
                                         np.array([1., 0., -1., 0.7]), 2)
    np.testing.assert_array_almost_equal(model._get_sim_scores(2),
                                         np.array([-1., -1., 1., 0.7]), 2)


def test_top_k(model: UserMemoryModel, ratings: np.ndarray) -> None:
    model.fit(ratings)
    assert model.top_k_items(0, 3) == [0, 1, 2]
    assert model.top_k_items(2, 3) == [0, 3, 1]
