# Recommender systems

[![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/olekssy/recommender-primer)](Pipfile)
[![GitHub](https://img.shields.io/github/license/olekssy/recommender-primer)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/olekssy/recommender-primer)](https://github.com/olekssy/recommender-primer/commits/main)

A library of recommender systems with collaborative, content-based filtering, and hybrid models.

[Collaborative filtering](#collaborative-filtering)

* [User-based memory model](#user-based-memory-model)
* [Item-based memory model](#item-based-memory-model)

## Collaborative filtering

* Takes an arbitraty `m x n` rating matrix with discrete or continuous observed ratings.
* Computes similarity score matrix in the offline mode.
* Estimates user (item) similarity score as Pearson correlation, (adj) cosine similarity, z-score of mutually observed ratings.
* Predicts the rating of an item for a user as a weighted dot product of positive similarity scores and observed peer ratings.
* Tune model by fitting similarity sensitivity with alpha > 1, and min significant similarity score threshold.

### User-based memory model

A neighborhood-based collaborative filtering method for predicting the target item rating for a user from observed ratings of similar users.

```python
>>> from recommenders.collaborative import UserMemoryModel

>>> rating_matrix
array([[nan,  2.,  0., nan],
       [-2., nan, nan,  0.],
       [ 1., -1., nan, nan],
       [ 1.,  0., -1., nan]])

>>> umm = UserMemoryModel(sim_method='pearson',
                          alpha=1.0,
                          min_similarity=0.1)
```

Estimate similarity of user(1) to peers based on the mutually observed ratings.

```python
>>> umm.fit(rating_matrix)
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

Estimated similarity scores of users.

```python
>>> umm.similarity_scores.round(1)
array([[ 1. ,  0. , -1. ,  0.7],
       [ nan,  1. , -1. , -1. ],
       [ nan,  nan,  1. ,  0.7],
       [ nan,  nan,  nan,  1. ]])
```

---

### Item-based memory model

A neighborhood-based collaborative filtering method for predicting the target item rating for a user from observed ratings of similar users.

```python
>>> from recommenders.collaborative import ItemMemoryModel

>>> rating_matrix
array([[nan,  2.,  0., nan,  1., -1.],
       [-2., nan, nan,  0., nan,  1.],
       [ 1., -1., nan, nan,  0., nan],
       [ 1.,  0., -1., nan,  2., -2.]])

>>> imm = ItemMemoryModel(alpha=1.0, min_similarity=0.1)
```

Estimate similarity of item(0) to comparable items from observed ratings.
Note, unlike the user-based model, the item-based model estimates similarity scores between items (columns).
The element at 0-th index in the array is the 0-th item itself.

```python
>>> imm.fit(rating_matrix)
>>> imm.similarity(0).round(1)
array([ 1. , -0.7, -1. , -1. ,  0.7, -0.9])
```

Predict rating of item(3) for user(3).

```python
>>> imm.predict(3, 3)
-2.0
```

Predict top-3 rated items for user(3).

```python
>>> imm.top_k_items(3, 3)
[4, 0, 1]
```

Predict missing ratings for all users.
The item-based model requires more observed items for solving a matrix completion problem, compared to the user-based model.
Since the item-based model predicts the rating of a target item from the other rated items _by the same user_, it is believed to provide more personalized prediction, at a cost of required number of rated items by the target user.

```python
>>> imm.complete_rating_matrix()
array([[ 1.,  2.,  0., -1.,  1., -1.],
       [-2., nan,  1.,  0., -2.,  1.],
       [ 1., -1., nan, nan,  0., nan],
       [ 1.,  0., -1., -2.,  2., -2.]])
```

Estimated similarity scores of items.

```python
>>> imm.similarity_scores.round(1)
array([[ 1. , -0.7, -1. , -1. ,  0.7, -0.9],
       [ nan,  1. , -0.4,  0. ,  0.2, -0.6],
       [ nan,  nan,  1. ,  0. , -1. ,  1. ],
       [ nan,  nan,  nan,  1. ,  0. ,  1. ],
       [ nan,  nan,  nan,  nan,  1. , -0.9],
       [ nan,  nan,  nan,  nan,  nan,  1. ]])
```

## Dependencies

Install environment and dependencies with `pipenv sync --dev`

* `pipenv >= 2022.5.2`
* `pyenv >= 2.2.5`

## References

* Aggarwal, C. C. (2016), Recommender Systems - The Textbook, Springer.
