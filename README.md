## Problem Statement

- pbm statement, motivation, input, output
- what people have done? references


## Data

- explain given data, visualize
![alt text](https://github.com/pruthviperumalla/Event-Recommender-System/blob/master/results/wordcloud.png "Word Cloud")
![alt text](https://github.com/pruthviperumalla/Event-Recommender-System/blob/master/results/wordBarPlot.png "Word Distribution")




## Approach

Formalizing our problem as recommendation modelling and using techniques like collaborative filtering might not be a good idea for the following reasons. 
- For user/event based collaborative filtering model to be useful, there must be considerable overlapping of transactions between events and users which is not true in our case. The transactions data provided is too sparse for collaborating filtering to make useful recommendations. 
- Also, there are users and events that don't have an entry in the training data. 
- Another reason is that recommendations by collaborative filtering are generally generated from the open list of all events whereas the challenge requires us to generate recommendations for a user from the provided closed list of events.
- Custom features derived from the provided features of users and events may best determine the similarity between user and event and generate meaningful recommendations.

For the above reasons, we model this problem as a binary classification problem in which given a pair of user and event, we classify whether the user is interested in attending the event. Overall, the recommendation system can be divided into three phases- feature extraction, interest prediction and generation of recommendations. 

<!--The approach is to first extract features related to user, features related to event and custom features that measure the similarity between user and event based on the attedance history available. Then, use these features to learn supervised model that predicts if a user is interested in an event given.  -->


### Feature Extraction

In this phase, we perform feature engineering to identify and extract features that drive the prediction of user's interest in an event.  For a user and event pair, following are the features extracted. 
1. \[ number/ratio of users attending, not attending, maybe attending and invited to the event \]
  
2.  \[location similarity between user and event \]

3. \[time to event, apparently most important feature; \]
As we were mentioning before, one major drawback with event based recommendations is the time sensitivity of it. A user can only attend an event that is scheduled after reasonable amount of time and he/she cannot do anything about the past events or events that are starting very soon. Intuitively, the time difference between when the event is scheduled to start and when the user first came to know about it (in our case, time at which the user saw a notification about the event) is represented in this feature.

4. \[similarity between user and event based on attendance \]

5. Similarity between the user and the event based on cluster of words

6. Boolean indicating whether the user was invited to the event.
7. Boolean indicating whether the event was created by the user's friend. 
8. Gender of the user.
9. Age of the user determined based on the provided date of birth.

### Interest Prediction

### Generation of Recommendations
  
##  Experiments & Results
    
- list all models
- metric disucssion
- baseline models
- results plots
- analysis

## Conclusion and Future Work
- major achievement?
- future work - recommend events


