from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
import numpy as np

def fit(X_train, y_train, estimators, weights):
    # estimators=[(‘knn’, knn_best), (‘rf’, rf_best), (‘log_reg’, log_reg)]
    ensemble_model = VotingClassifier(estimators, voting='hard', weights = weights) #'hard'
    metric = make_scorer(fbeta_score, beta=0.5)
    scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring = metric)
    print(scores)
    print("avg kfold val f0.5 score", np.average(scores))
    return ensemble_model

def fit_predict(X_train, y_train, X_test, ensemble_model):
    ensemble_model.fit(X_train, y_train)
    y_pred = ensemble_model.predict(X_train)
    print("training accuracy: ", accuracy_score(y_train, y_pred))
    print("training f0.5 score: ", fbeta_score(y_train, y_pred, beta = 0.5))
    return ensemble_model.predict(X_test)