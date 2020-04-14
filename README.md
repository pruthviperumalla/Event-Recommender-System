# Event Recommendation Engine


## Problem Statement

- pbm statement, motivation, input, output
- what people have done? references


## Data

- explain given data, visualize

## Approach

Formalizing our problem as recommendation modelling and using techniques like collaborative filtering might not be a good idea for the following reasons. 
- For user/event based collaborative filtering model to be useful, there must be considerable overlapping of transactions between events and users which is not true in our case. The transactions data provided is too sparse for collaborating filtering to make useful recommendations. 
- Also, there are users and events that don't have an entry in the training data. 
- Another reason is that recommendations by collaborative filtering are generally generated from the open list of all events whereas the challenge requires us to generate recommendations for a user from the provided closed list of events.
- Several features of events and users are provided which can be used to extract custom features.

For the above reasons, we model this problem as a binary classification problem in which given a pair of user and event, we classify whether the user is interested in attending the event. The recommendation engine can be divided into following phases of feature extraction, interest prediction and generation of recommendations. 


<!--The approach is to first extract features related to user, features related to event and custom features that measure the similarity between user and event based on the attedance history available. Then, use these features to learn supervised model that predicts if a user is interested in an event given.  -->


### Feature Extraction
- explain custom features, visualizations
  
### Interest Prediction

### Generation of Recommendations
  
##  Experiments & Results
    
- list all models
- metric disucssion
- baseline models
- results plots
- analysis

## Conclusion
- major achievement?
- future work - recommend events


