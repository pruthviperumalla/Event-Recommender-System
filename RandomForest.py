from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score 

def fit(X_train, y_train):
    model = RandomForestClassifier(n_estimators = 10, max_depth = 10)
    # model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(scores)
    return model

def fit_predict(X_train, y_train, X_test, model):
    model.fit(X_train, y_train)
    return model.predict(X_test)