# Recommender systems

[![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/olekssy/recommender-primer)](Pipfile)
[![GitHub](https://img.shields.io/github/license/olekssy/recommender-primer)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/olekssy/recommender-primer)](https://github.com/olekssy/recommender-primer/commits/main)

A library of recommender systems with collaborative, content-based filtering, and hybrid models.

## Collaborative filtering

### User-based memory model

A neighborhood-based collaborative filtering method for predicting the target item rating for a user from observed ratings of similar users.

* Takes an arbitraty `m x n` rating matrix with discrete or continuous observed ratings.
* Estimates user similarity score as Pearson correlation (cosine similarity, z-score) of mutually observed item ratings.
* Predicts the rating of an item for a user as a weighted dot product of positive similarity scores and observed peer ratings.

```python
>>> from recommenders.collaborative import UserMemoryModel
```

Define `m_users x n_items` observed ratings matrix.

```python
>>> rating_matrix
array([[nan,  2.,  0., nan],
       [-2., nan, nan,  0.],
       [ 1., -1., nan, nan],
       [ 1.,  0., -1., nan]])
```

Set similarity method (cosine, pearson, z-score), h-parameters.

```python
>>> umm = UserMemoryModel(sim_method='pearson',
                          alpha=1.0,
                          min_similarity=0.1)
```

Fit model to observed ratings.

```python
>>> umm.fit(rating_matrix)
```

Estimate similarity of user(1) to peers.

```python
>>> umm.similarity(user_id=1)
array([ 0.,  1., -1., -1.])
```

Predict rating of item(0) for user(0).

```python
>>> umm.predict(user_id=0, item_id=0)
2.0
```

Predict top-3 rated items for user(0).

```python
>>> umm.top_k_items(user_id=0, k=3)
[0, 1, 2]
```

Predict missing ratings for all users.
Note, the model fails to predict ratings for some user-item pairs due to the sparsity of observed ratings and lack of similar peers.

```python
>>> umm.complete_rating_matrix()
array([[ 2.,  2.,  0., nan],
       [-2., nan, nan,  0.],
       [ 1., -1., -1., nan],
       [ 1.,  0., -1., nan]])
```

Estimated similarity scores of users

```python
>>> umm.similarity_scores.round(1)
array([[ 1. ,  0. , -1. ,  0.7],
       [ nan,  1. , -1. , -1. ],
       [ nan,  nan,  1. ,  0.7],
       [ nan,  nan,  nan,  1. ]])
```

## Dependencies

Install environment and dependencies with `pipenv sync --dev`

* `pipenv >= 2022.5.2`
* `pyenv >= 2.2.5`
