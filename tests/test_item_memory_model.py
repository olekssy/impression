import pytest
import numpy as np
from recommenders.collaborative_filtering import ItemMemoryModel


@pytest.fixture
def ratings() -> np.ndarray:
    return np.array([
        [np.nan, 2, 0, np.nan, 1, -1],
        [-2, np.nan, np.nan, 0, np.nan, 1],
        [1, -1, np.nan, np.nan, 0, np.nan],
        [1, 0, -1, np.nan, 2, -2],
    ])


@pytest.fixture
def model() -> ItemMemoryModel:
    return ItemMemoryModel()


def test_hparam_alpha(model: ItemMemoryModel, ratings: np.ndarray) -> None:
    model.alpha = 2.0
    model.fit(ratings)
    assert round(model.predict(0, 0)) == 0.0
    assert round(model.predict(2, 2), 1) == 0.5


def test_predict(model: ItemMemoryModel, ratings: np.ndarray) -> None:
    model.fit(ratings)
    assert model.predict(0, 0) == 1.0
    assert model.predict(1, 2) == 1.0
    assert model.predict(3, 3) == -2.0


def test_mu(model: ItemMemoryModel, ratings: np.ndarray) -> None:
    expected = np.array([0.5, -0.3, 0., 0.])
    model.fit(ratings)
    np.testing.assert_array_almost_equal(model.mu, expected, 1)


def test_similarity(model: ItemMemoryModel, ratings: np.ndarray) -> None:
    expected = np.array([1.0, -0.7, -1.0, -1.0, 0.7, -0.9])
    model.fit(ratings)
    np.testing.assert_array_almost_equal(model.sim_scores[0], expected, 1)


def test_top_k(model: ItemMemoryModel, ratings: np.ndarray) -> None:
    model.fit(ratings)
    assert model.top_k_items(0, 3) == [1, 0, 4]
    assert model.top_k_items(3, 3) == [4, 0, 1]
