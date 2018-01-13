# MEBoost: Mixing Estimators with Boosting for Imbalanced Data Classification

# Abstract 
Class imbalance problem has been a challenging
research problem in the fields of machine learning and data
mining as most real life datasets are imbalanced. Several existing
machine learning algorithms try to maximize the accuracy
classification by correctly identifying majority class samples while
ignoring the minority class. However, the concept of the minority
class instances usually represents a higher interest than the
majority class. Recently, several cost sensitive methods, ensemble
models and sampling techniques have been used in literature in
order to classify imbalance datasets. In this paper, we propose
MEBoost, a new boosting algorithm for imbalanced datasets.
MEBoost mixes two different weak learners with boosting to
improve the performance on imbalanced datasets. MEBoost is
an alternative to the existing techniques such as SMOTEBoost,
RUSBoost, Adaboost, etc. The performance of MEBoost has
been evaluated on 12 benchmark imbalanced datasets with state
of the art ensemble methods like SMOTEBoost, RUSBoost,
Easy Ensemble, EUSBoost, DataBoost. Experimental results show
significant improvement over the other methods and it can be
concluded that MEBoost is an effective and promising algorithm
to deal with imbalance datasets

Paper here: https://arxiv.org/pdf/1712.06658.pdf

Use fit() to train predict() to get predictions. predict_proba() and predict_proba_samme() can be used to get probabilites.

Please cite our paper if our paper or code helped you.
