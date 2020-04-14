from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
import numpy as np

def fit(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors = 20, weights = 'distance', algorithm = 'auto')
    # model.fit(X_train, y_train)
    metric = make_scorer(fbeta_score, beta=0.5)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring = metric)
    print(scores)
    print("avg kfold val f0.5 score", np.average(scores))
    return model

def fit_predict(X_train, y_train, X_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print("training accuracy: ", accuracy_score(y_train, y_pred))
    print("training f0.5 score: ", fbeta_score(y_train, y_pred, beta = 0.5))
    return model.predict(X_test)