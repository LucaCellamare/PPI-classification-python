import utils
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import SGDClassifier,LogisticRegression


# CROSS-VALIDATION ACCURACY

def validation_accuracy(X, labels):
    kfolds = KFold(n_splits=10)
    sgd = SGDClassifier(loss="log",penalty='l1', max_iter=300, tol=1e-3,class_weight="balanced")
    model = OneVsRestClassifier(sgd, n_jobs=1)
    scores = cross_val_score(model, X, labels, cv=kfolds, n_jobs=32, verbose=2, scoring="f1_micro")
    print(f"Mean crossvalidation Micro-F1: {np.mean(scores):.3f}")


# TRAINING ACCURACY AND PREDICTION

def fit_model(X, labels):
    sgd = SGDClassifier(loss="log", max_iter=300,tol=1e-3,class_weight="balanced")
    model = OneVsRestClassifier(sgd, n_jobs=1)
    model.fit(X, labels)
    train_pred = model.predict_proba(X)
    train_pred = train_pred > 0.4
    #train_pred = model.predict(X)
    utils.get_score(train_pred, labels)
    return model

