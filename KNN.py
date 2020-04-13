from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer

def fit(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors = 30, weights = 'distance', algorithm = 'auto')
    # model.fit(X_train, y_train)
    metric = make_scorer(fbeta_score, beta=0.5)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring = metric)
    print(scores)
    return model

def fit_predict(X_train, y_train, X_test, model):
    model.fit(X_train, y_train)
    return model.predict(X_test)