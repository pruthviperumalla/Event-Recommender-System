## Introduction

Proliferation of the amount of data available in every sector has created an increasing demand for machine learning based systems including recommender systems and they have become quite ubiquitous. Slight variations of recommender systems are currently in use across multiple industries. These systems try to recommend new items to customers/users based on their past preferences and experiences. For example, new restaurants are recommended for someone using a restaurant aggregator application or new products to buy for a customer on an e-commerce application. In this project, we primarily focus on Event recommendation systems that traditionally need different algorithms [1] from an item recommender that recommends books or movies. We explore what techniques best work for event recommendations.

## Dataset

We initially started with an idea to use some practical data from [Atlanta Habitat](http://www.atlantahabitat.org/) but some unexpected delays in obtaining the data made fall back upon a publicly availabe dataset. The [Kaggle Events Dataset](https://www.kaggle.com/c/event-recommendation-engine-challenge) that we used was from an events related anonymous application that hosted a public contest a few years ago. It was spread across different files giving us the choice to pick up custom features and build on top of it. Here is a summary of the dataset components that we used:

1. **Events** - This had data about 3137972 events related information with an event id, event creator id, event start time and event location details such as city, state, country, zip, latitude, longitude. Along with this, top 100 most occuring words from all of the event descriptions are taken and their frequency checked in each of the descriptions individually. These 100 count values are provided for each of the event. Notice the missing location data in the figure below.

   Figure: Missing values in events data
   ![alt text](./results/GreenEventMSNO.png "Missing values in events data")

   Figure: Frequency of occurence of top 50 representative words
   ![alt text](./results/wordBarPlot.png "Word Distribution")

2. **Users** -  Here, we have data about 38209 users in the form of their ids, user locale, birthyear, gender, timestamp at which user joined, location and timezone. Notice the missing location data in the figure below.

   Figure: Missing values in users data
   ![alt text](./results/GreenUserMSNO.png "Missing values in users data")

3. **User friends** - For each of the user id mentioned in Users, we have a list of user ids who are his/her friends. We only have 38202 user id here though.

   Figure: Any graphs on friends

4. **Event attendees** -  For some of the events mentioned in Events (24144 events to be precise), we have fours user id lists
            - "yes" gives us the list of users who marked as going for this event
            - "maybe" gives us the users who marked as maybe going
            - "no" gives us the list of users who marked as not going
            - "invited" gives us the list of users who were invited to the event
            
5. **User-Event interests** - For 15398 event-user pairs, we have information on whether this user was invited to the event, timestamp at which this user saw the notification for this event and also two boolean values indicating whether this user markers "interested" or "not interested" for the event.

## Approach

Traditional recommender system algorithms such as collaborative filtering would not work well for event recommendations [1]. The following reasons explain some of the reasoning behind not formalizing our problem as recommendation modelling.

- **Sparse collaborative information:** For user/event based collaborative filtering model to be useful, there must be considerable overlapping of transactions between events and users which is not true in our case. The transactions data provided is too sparse for collaborating filtering to make useful recommendations. 
- **New item problem:** Fundamentally, an event is very different from a book or movie consumption as there would be no consumption before the event occurs and it cannot be reused. It could be a similar kind of event but some of it's features such as start time and attendance list would be different. 
- **Limited training data:** In our case specifically, there are users and events that don't have an entry in the training data. 
- **Available custom features**: Custom features derived from the provided features of users and events may best determine the similarity between user and event and generate meaningful recommendations.

For the above reasons, we model this problem as a binary classification problem in which given a pair of user and event, we classify whether the user is interested in attending the event. Overall, the recommendation system can be divided into three phases- feature extraction, interest prediction and generation of recommendations. We had to spend most of our time on feature extraction in this experiment.

<!--The approach is to first extract features related to user, features related to event and custom features that measure the similarity between user and event based on the attedance history available. Then, use these features to learn supervised model that predicts if a user is interested in an event given.  -->


### 1. Feature Extraction

  In this phase, we perform feature engineering to identify and extract features that drive the prediction of user's interest in an event.  For a user and event pair, following are the features extracted. 

1. ***Event attendance based metrics***

   There is a higher chance of our target user attending an event that is already popular. To measure the popularity of an event, we extract the following metrics based on user attendance and set up various features using these. 
   
      - Number of users who marked "attending" for our event
      - Number of users who marked "no attending" for our event
      - Number of users who marked "may be" for our event
      - Number of users who are invited to the event
      - Ratio of number of users not attending with number of users attending
      - Ratio of number of users who might be attending with number of users attending
      - Ratio of number of users invited with number of users attending

2. ***Friend's attendance based metrics***

   Building on top of the first feature, looking at only the attendance of current user's friends might give us a stronger measure of current user's interests with that of the event. Hence, we extracted a similar set of features resitrcting ourselves to only look at the friends of current user instead of everyone. The following features are extracted in this way. 
   
      - Number of friends who marked "attending" for our event
      - Number of friends who marked "no attending" for our event
      - Number of friends who marked "may be" for our event
      - Number of friends who are invited to the event
      - Ratio of number of friends not attending with number of friends attending
      - Ratio of number of friends who might be attending with number of friends attending
      - Ratio of number of friends invited with number of friends attending

3. ***Local convenience***

   Users are more likely to attend an event that is happening nearby and one that is preferably in the same city. From the user and event details, we extract two boolean features. The first one checks if the user and event are in the same city and the second one checks if they are in the same country. For a better understanding we plotted the all the city locations where events are happening and where the user are located in a world map. 
   
   <div align="center" style="align-items: center;justify-content: center;">
    <img  src="./results/location_events.png" width="650" >
   </div>
   <div align="center">
    <i>Event cities from training data scaled by number of events in a city</i>
   </div>
   
   <div align="center" style="align-items: center;justify-content: center;">
    <img  src="./results/location_users.png" width="650">
   </div>
   <div align="center">
    <i>User city locations scaled by number of users in a city</i>
   </div>

4. ***Schedule convenience***

   As we were mentioning before, one major drawback with event based recommendations is the time sensitivity of it. A user can only attend an event that is scheduled after reasonable amount of time and he/she cannot do anything about the past events or events that are starting very soon. Intuitively, the time difference between when the event is scheduled to start and when the user first came to know about it (in our case, time at which the user saw a notification about the event) is represented in this feature.

   <div align="center" style="align-items: center;justify-content: center;">
    <img  src="./results/time_diff.png">
   </div>
   <div align="center">
    <i>Time difference between event start time and user notification time</i>
   </div>

5. ***Event similarity based on previously attended events***

   Analysing the common attenders between two events might give us hints on how likely a user will attend the second event given that he has attended the first event. Higher number of common attenders would indicate higher similarity between the events and higher likeliness to attend to both the events. In this feature we try to model this event similarity by looking at the common attenders between our current target event and the events our current user has previously attended. For each such pair, we normalize the number of intersecting users by dividing them with the number of attendees from the lower attended event. The average of all such pairs is taken as our similarity score.

6. ***Sometimes, people just prefer the familiar***
   
   If a user is interested in events related to a topic, they may also attend future events related to this topic. We can capture this feature by measuring the similarity between the event in question to the events user attended in the past. For every event, we are also provided a bag of top 100 frequent words constructed from its name and description. We use this bag of words features to cluster together the events. To perform the unsupervised clustering, we use K-Means algorithm. Figure <> shows the elbow curve that is used to determine the optimal number of clusters. However, the elbow is not obvious from the plot. As the total number of events is extremely high (3 million), we chose 200 clusters which is a trade-off between computation time and loss (sum of squared distances).

   <!-- ![alt text](./results/elbow.png "Word Distribution") -->

   <div align="center">
   <img  src="./results/elbow.png"> </div>
   <div align = "center"><i>Figure 2: K Means Elbow Plot </i> </div>

   To visualize the clusters, we reduced the dimensions from 100 to 3 using PCA. Below is an interactive visualization of the first three principal components of events and their clusters.  

   <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://pruthviperumalla.github.io/Event-Recommender-System/cluster.html" height="525" width="100%"></iframe>
   
   Using these clusters, we extract four similarity features one for each of the different interactions user had with the past events. i.e. For the first feature, we measure the similarity between the current event and the events for which user indicated that they were going. Second feature captures the similarity between the event in question to the events which user indicated that they maybe going. Third, similarity is measured between current event and the events for which user indicated that they were not going. Fourth feature captures the similarity between the current event and the events to user was invited. To measure the similarity between an event *e* and a list of events, we take the average of the euclidean distance  between the centroid of the cluster *e* belongs to and the centroids of the clusters of events from the list. 

7. ***Sometimes, people just like what their friends like***

   Users may also attend events whose topics are similar to those that their friends attended. We capture three more features similar to the above features. Specifically, we extract similarity between the current event and the events that user's friends indicated that they were going, maybe going and not going as three different features. 

8. ***Sometimes, people just prefer the events they were invited to*** 

   This is a boolean feature indicating whether the user was invited to the event.

9. ***Sometimes, people just prefer the events created by their friends***
   
   This is a boolean feature indicating whether the event was created by the user's friend. 

10. ***Some events may be gender specific***
   
      We capture the gender of the user that may be a determining factor for events that are either gender-specific or gender-related.

11. ***Age influences social goals*** 
      
      Age may be an important feature considering that the social goals of humans vary with age. We extract this feature from the year of birth provided for a user.
      
      Below is the heatmap for the correlation between the features. 
      
   <div align="center">
   <img  src="./results/heatmap.png"> </div>
   <div align = "center"><i>Figure 3: Correlation Matrix</i> </div>
      
- how missing values are handled

### 2. Interest Prediction
In this phase, we use the above extracted features to learn a classifier that predicts if a user is interested in a given event. We experimented with several supervised binary classification models with interested and not interested as the classes. Experiments performed with each of these models and results obtained are discussed in detail in the next section.

### 3. Generation of Recommendations
To generate recommendations for a user, we consider every event from the given closed list and predict if user is interested in it. The list of events that the system classifies as interested are then recommended to the user.

##  Experiments & Results

The number of traning samlpes that we have from each of the class is represented below. 

Figure of class imbalance.

This shows us there is a class imbalance where the number of examples from "not interested" class is almost three times the number of samples from interested class. In such situations of class imbalance, a naive classifier that always predicts a "not interested" class would yield very high accuracies that would sometimes be greater than our trained model's accuracy. 

Therefore, accuracy should not be the only metric to interpret the results of various models and we looked for other baseline 
metrics such as F measure. In our current use case, we would use our class prediction to send out event invites to users who would potentitally attend. Since we do not want to spam all the users, we want the invites to more precise and we have chosen F0.5 measure specifcally as it gives more importance to precision than recall.

F0.5 score is given by the following formula


### Baseline strategy:

1. **Accuracy:** For accuracy, the baseline strategy would be to always predict the most frequent class as that would yield the highest possible accuracy.

2. **F0.5 score:** Precision and recall always counter balance each other. Predicting the minority class (not interested in our case) would give a good lower 
bound for F0.5 score [2]. 


Our train test split is 80:20. To avoid overfitting and tune the hyperparameters, we used 5-folds cross validation on the training split. Below, we discuss the experiments, results and analysis of the various models we trained in the interest prediction phase. 

- **Random Forest**

   After tuning hyper parameters using k folds cross validation, we got the best results for an ensemble of 100 trees with depth limited to 15, gini impurity as the criterion to split, and with bootstrapping of the total number of data points drawn as samples and square root of the total number of features considered for the construction of a tree. The relative feature importances determined by the Random Forest model are shown in the figure <#>.

   <div align="center">
   <img  src="./results/rf_feat_imp.png"> </div>

   <div align = "center"> <i> Figure 3: Relative Feature Importances </i> </div>

   As seen from the figure above, time to event is the most important feature that determines whether a user will be interested in attending an event or not. The peformance of the Random Forest is the best out of all supervised classifiers we tried. 

- **K Nearest Neighbors**

   In this model, we used 20 nearest neighbors for classification where each point is weighted by the inverse of their distance. KNN also produced above baseline results.

- **Logistic Regression**

   In logistic regression, we first used a standard scaler to normalize all the features to remove the dominance of higher magnitude features such as "Schedule convenience". In the actual model, we used a l2 regularizer with a regularization strength of 1.0 to train. We also tried tuning the regularization strength hyperparameter but it was not effecting the validation accuracy significantly.

- **Neural Network**

   We used a simple fully connected multi layer perceptron to train our network. To decide on the number of hidden layers and number of nodes in each layer, we experimented with various values and started with the intuitive rule that the number of nodes in a hidden layer should be between the number of input nodes (31, in our case) and the number of output nodes(2, in our case).
   
   | No. of nodes in each hidden layer | Accuracy  | Fbeta Score  |
   | 15 | 0.736038961038961  | 0.41358293426208087  |
   | 20, 10 | 0.7126623376623377  | 0.396498455200824  |
   | 25 | 0.7337662337662337  | 0.4446038006453926  |
   | 30 | 0.7243506493506493  | 0.4235211760588029  |
   
- **Support Vector Machine**

   For Support Vector Machine, we tuned the hyperparameter C which controls the regularization strength of the model. We found out the SVM gives the best performance when the regularization is set to around 2000. We also tried different kernels such as "Radial Basis Kernel", "Linear Kernel" and "Polynomial Kernel" and the model has the highest test accuracy when using "Radial Basis Kernel".

- **Gaussian Naive Bayes**

   Peformance of Gaussian Naive Bayes model is below the baseline model in terms of F0.5 score. The naive assumption that the every pair of features is conditionally independent may not hold in our case. Also, Naive Bayes gives equal importance to all the fields and does not discrimate betweeen the features. These may be reasons for its low performance.

- **Ensemble Classifier**

   To further improve the performance, we experimented with an ensemble of several of the above models we implemented. We obtained the best validation as well as the test results with an ensemble of Random Forest classifier and KNN classifer. These two classifiers are ensembled by the hard(majority) voting mechanism with 0.9 weight for Random Forest and 0.1 weight for KNN. In terms of F0.5 score, the ensemble model shows 2.6% improvement over the Random Forest classifier and 4.8% improvement over the KNN classifier.

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

1. [Event Recommendation in Event-based Social Networks](http://ceur-ws.org/Vol-1210/SP2014_02.pdf)
2. [What Is the Naive Classifier for Each Imbalanced Classification Metric?](https://machinelearningmastery.com/naive-classifiers-imbalanced-classification-metrics/)

