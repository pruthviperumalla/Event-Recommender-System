from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer

def fit(X_train, y_train):
    metric = make_scorer(fbeta_score, beta=0.5)
    
    pipe =  DummyClassifier(random_state=0, strategy="constant", constant=1)
    
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring = metric)
    print("F0.5 Cross Validation Scores: ", scores)
    return pipe

def fit_predict(X_train, y_train, X_test, model):
    model.fit(X_train, y_train)
    return model.predict(X_test)


