from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
import numpy as np

# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("Recall: ", recall_score(y_test, y_pred))
# print("F0.5: ", fbeta_score(y_test, y_pred, beta = 0.5))
# print("Precision: ", precision_score(y_test, y_pred))
# print("F1: ", f1_score(y_test, y_pred))
# print("Confusion matrix", confusion_matrix(y_test, y_pred))

def fit(X_train, y_train):
    X_train.same_city = X_train.same_city.astype(int)
    X_train.same_country = X_train.same_country.astype(int)
    X_train.is_creator_friend = X_train.is_creator_friend.astype(int)
    
    metric = make_scorer(fbeta_score, beta=0.5)
    
    pipe = make_pipeline( StandardScaler(),
                     MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25), random_state=1, max_iter = 10000)
                     )
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring = metric)
    print("F0.5 Cross Validation Scores: ", scores)
    print("avg kfold val f0.5 score", np.average(scores))
    return pipe

def fit_predict(X_train, y_train, X_test, model):
    X_train.same_city = X_train.same_city.astype(int)
    X_train.same_country = X_train.same_country.astype(int)
    X_train.is_creator_friend = X_train.is_creator_friend.astype(int)
    
    X_test.same_city = X_test.same_city.astype(int)
    X_test.same_country = X_test.same_country.astype(int)
    X_test.is_creator_friend = X_test.is_creator_friend.astype(int)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print("training accuracy: ", accuracy_score(y_train, y_pred))
    print("training f0.5 score: ", fbeta_score(y_train, y_pred, beta = 0.5))
    return model.predict(X_test)



