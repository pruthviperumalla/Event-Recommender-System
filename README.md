## Introduction

Recommender System is one of the growing demands in the current situation where the amount of information available on the internet is humongous. Recommender Systems can be useful for enterprises and directly for the users so as to decide what is best for them. The recommendation will be based on the past preferences and experiences so as to increase the probability of recommending right things. Recommender systems can recommend items based on the available information, for example, recommending restaurants based on the past dining out options or the available restaurant around a given location. It helps the enterprises (here restaurant owners) as the recommendation system could become an additional source which brings more business. 

## Objective
These recommendations can be extended to books, movies, organised profitable events (like conferences or music festivals) or non-profitable events (such as run for charity or fund raising activities). Recommending these events to the user based on their past experiences and preferences is one of the basic motivations to develop an event recommendation system. This recommendation takes into account varied amounts of information about the users and the events. Information related to the user includes age, gender etc. For events, the information includes the description, type, past attendees, location of the event etc. Apart from this, the system needs some information which relates users to events, like distance between the location user and event, or information if any user’s friends are visiting/organising the event. 	

This project tries to recommend whether the user would be interested in the given events (with details about type, location) based on the information available about the users, events and user's friends. 

We have used the dataset available on the Kaggle website [1:https://www.kaggle.com/c/event-recommendation-engine-challenge] and the competition was opened 7 years ago hence there are variety of solutions available for this dataset. 


## Data

The data available on the Kaggle website had huge dataset available. However, the data was not perfect that is it had missing values for several fields. Image below shows the missing values in the location field for users.csv available.


![alt text](./results/GreenUserMSNO.png "Missing values in users data")

 Similarly, the dataset available for events had missing values for various fields as shown in the below image. 
 
 
![alt text](./results/GreenEventMSNO.png "Missing values in events data")

- explain given data, visualize
![alt text](./results/wordcloud.png "Word Cloud")
![alt text](./results/wordBarPlot.png "Word Distribution")
*Fig. 2: Title*

- how missing values are handled

## Approach

Formalizing our problem as recommendation modeling and using techniques like collaborative filtering might not be a good idea for the following reasons. 
- For user/event based collaborative filtering model to be useful, there must be considerable overlapping of transactions between events and users which is not true in our case. The transactions data provided is too sparse for collaborating filtering to make useful recommendations. 
- Also, there are users and events that don't have an entry in the training data. 
- Another reason is that recommendations by collaborative filtering are generally generated from the open list of all events whereas the challenge requires us to generate recommendations for a user from the provided closed list of events.
- Custom features derived from the provided features of users and events may best determine the similarity between user and event and generate meaningful recommendations.

For the above reasons, we model this problem as a binary classification problem in which given a pair of user and event, we classify whether the user is interested in attending the event. Overall, the recommendation system can be divided into three phases- feature extraction, interest prediction and generation of recommendations. 

<!--The approach is to first extract features related to user, features related to event and custom features that measure the similarity between user and event based on the attedance history available. Then, use these features to learn supervised model that predicts if a user is interested in an event given.  -->


### 1. Feature Extraction

In this phase, we perform feature engineering to identify and extract features that drive the prediction of user's interest in an event.  For a user and event pair, following are the features extracted. 

1. \[ number/ratio of users attending, not attending, maybe attending and invited to the event \]

2. \[ number/ratio of users' friends attending, not attending, maybe attending and invited to the event \]

3.  \[location similarity between user and event \]

4. \[time to event, apparently most important feature; \]
As we were mentioning before, one major drawback with event based recommendations is the time sensitivity of it. A user can only attend an event that is scheduled after reasonable amount of time and he/she cannot do anything about the past events or events that are starting very soon. Intuitively, the time difference between when the event is scheduled to start and when the user first came to know about it (in our case, time at which the user saw a notification about the event) is represented in this feature.

5. \[similarity between user and event based on attendance \]

6. *Sometimes, people just prefer the familiar*
   
   If a user is interested in events related to a topic, they may also attend future events related to this topic. We can capture this feature by measuring the similarity between the event in question to the events user attended in the past. For every event, we are also provided a bag of top 100 frequent words constructed from its name and description. We use this bag of words features to cluster together the events. To perform the unsupervised clustering, we use K-Means algorithm. Figure <> shows the elbow curve that is used to determine the optimal number of clusters. However, the elbow is not obvious from the plot. As the total number of events is exteremely high (3 million), we chose 200 clusters which is a trade-off between computation time and loss (sum of squared distances).

   ![alt text](./results/elbow.png "Word Distribution")
   
   <div align="center"> <i>Figure 2: K Means Elbow Plot </i></div>

   To visualize the clusters, we reduced the dimensions from 100 to 3 using PCA. Below is an interactive visualization of the first three principal components of events and their clusters.  

   <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://pruthviperumalla.github.io/Event-Recommender-System/cluster.html" height="525" width="100%"></iframe>

   Using these clusters, we extract four similarity features one for each of the different interactions user had with the past events. i.e. For the first feature, we measure the similarity between the current event and the events for which user indicated that they were going. Second feature captures the similarity between the event in question to the events which user indicated that they maybe going. Third, similarity is measured between current event and the events for which user indicated that they were not going. Fourth feature captures the similarity between the current event and the events to user was invited. To measure the similarity between an event *e* and a list of events, we take the average of the euclidean distance  between the centroid of the cluster *e* belongs to and the centroids of the clusters of events from the list. 

7. *Sometimes, people just like what their friends like*

   Users may also attend events whose topics are similar to those that their friends attended. We capture three more features similar to the above features. Specifically, we extract similarity between the current event and the events that user's friends indicated that they were going, maybe going and not going as three different features. 

8. Boolean indicating whether the user was invited to the event.

9. Boolean indicating whether the event was created by the user's friend. 

10. Gender of the user.

11. Age of the user determined based on the provided year of birth.

### 2. Interest Prediction
In this phase, we use the above extracted features to learn a classifier that predicts if a user is interested in a given event. We experimented with several supervised binary classification models with interested and not interested as the classes. Experiments performed with each of these models and results obtained are discussed in detail in the next section.

### 3. Generation of Recommendations
To generate recommendations for a user, we consider every event from the given closed list and predict if user is interested in it. The list of events that the system classifies as interested are then recommended to the user.

##  Experiments & Results

- metric definiton, why 
- baseline models, why

Our train test split is 80:20. To avoid overfitting and tune the hyperparameters, we used 5-folds cross validation on the training split. Below, we discuss the experiments, results and analysis of the various models we trained in the interest prediction phase. 

- **Random Forest**
The relative feature importances determined by the random forest model are shown in the figure <#>.
![alt text](./results/rf_feat_imp.png "Feature Importances")

- **K Nearest Neighbors**

- **Logistic Regression**

- **Neural Network**

- **Support Vector Machine**

   For Support Vector Machine, we tuned the hyperparameter C which controls the regularization strength of the model. We found out the SVM gives the best performance when the regularization is set to around 2000. We also tried different kernels such as "Radial Basis Kernel", "Linear Kernel" and "Polynomial Kernel" and the model has the highest test accuracy when using "Radial Basis Kernel".

- **Gaussian Naive Bayes**

   Gaussian Naive Bayes model gave the following result. We could not tune the parameters as there were no hyperparameters present. This was one of the reasons which discouraged us to use this method for classification. Naive Bayes gives equal importance to all the fields and does not discrimate betweeen the features. 

- **Ensemble Classifier**

- VALUES NEED TO BE UPDATED!

| Model | Accuracy  | Fbeta Score  |
| :---:   | :-: | :-: |
| Gaussian Naive Bayes | 0.7204545454545455 | 0.2902155887230514 |
| Random Forest | 0.7746753246753246 | 0.5323383084577115 |
| KNN | 0.7616883116883116 | 0.5085435313262815 |
| SVM | 0.699025974025974 | 0.3810930576070901 |
| Neural Net | 0.7327922077922078 | 0.4452296819787986 |
| Logistic Regression | 0.7334415584415584 | 0.17395182872435327 |
| El | 0.7772727272727272 | 0.5445304937076476 |
| Ensemble Learning | 0.775974025974026 | 0.5396902226524685 |


- results plots
- analysis

## Conclusion and Future Work

We took the approach of experimenting and comparing a few supervised models to determine which technique works best to predict what events users will be interested in based on events they've responded to in the past, user demographic information, and what events they've interacted with in the app. Our current results show above baseline performance (F0.5 score) for Random Forest, KNN, Neural Network and SVM with Random Forest performing the best out of all. Experiments also show that an ensemble model of Random Forest and KNN further improves the F0.5 score by 2.6%. We conclude that the hand-crafted custom features to measure similarity between user and event perform reasonably well on the task of user interest prediction even though the transactions between users and events are sparse.

As an extension to the project, we would like to rank the generated event recommendations based on the probability of user interest predicted by the models. Another extension would be to employ a new feature of user clustering by modeling friendships between users as a graph and detecting communities.

## References


