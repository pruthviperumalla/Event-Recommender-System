from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import fbeta_score, make_scorer

def fit(X_train, y_train, estimators):
    # estimators=[(‘knn’, knn_best), (‘rf’, rf_best), (‘log_reg’, log_reg)]
    ensemble_model = VotingClassifier(estimators, voting='hard', weights = [0.8, 0.2]) #'hard'
    metric = make_scorer(fbeta_score, beta=0.5)
    scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring = metric)
    print(scores)
    return ensemble_model

def fit_predict(X_train, y_train, X_test, ensemble_model):
    ensemble_model.fit(X_train, y_train)
    return ensemble_model.predict(X_test)