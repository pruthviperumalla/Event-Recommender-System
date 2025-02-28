from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
# dataPath = "./data/"
# X_train = pd.read_csv(dataPath + 'X_train.csv')
# X_test = pd.read_csv(dataPath + 'X_test.csv')
# y_train = pd.read_csv(dataPath + 'y_train.csv')
# y_test = pd.read_csv(dataPath + 'y_test.csv')

# pipe = make_pipeline( StandardScaler(),
#                      SVC(kernel='rbf', gamma='auto')
#                      )

# cross_scores = cross_val_score(pipe, X_train, y_train, cv=5)
# print("Cross Validation Scores: ", cross_scores)

# pipe.fit(X_train, y_train)
# y_pred = pipe.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("Precision: ", precision_score(y_test, y_pred))
# print("Recall: ", recall_score(y_test, y_pred))
# print("F1: ", f1_score(y_test, y_pred))

# print("F0.5: ", fbeta_score(y_test, y_pred, beta = 0.5))
def fit(X_train, y_train):
    model = make_pipeline(StandardScaler(),
                          SVC(C=2000, kernel='rbf')
                     )
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