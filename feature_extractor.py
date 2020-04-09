import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import datetime as dt

def is_event_creator_friend(row):
    if (type(row['friends']) != str):
        return False
    
    if row['creator_id'] in row['friends'].split():
        return True
    else:
        return False
    
def time_to_event(row):
    start_time = dt.datetime.strptime(row['start_time'][:19], "%Y-%m-%dT%H:%M:%S")
    not_time = dt.datetime.strptime(row['timestamp'][:19], "%Y-%m-%d %H:%M:%S")
    diff_time = not_time - start_time
    return diff_time.total_seconds()

def add_event_creator_friend_timestamp_feature(train_df, events_df, friends_df):
    events_df = events_df[['event_id', 'user_id', 'start_time']]
    merged_df = pd.merge(train_df, events_df, how='inner', left_on='event', right_on='event_id', suffixes=('', '_event'))
    
    merged_df = merged_df.rename(columns={'user_id': 'creator_id'})
    merged_df2 = pd.merge(merged_df, friends_df, how='inner', left_on='user', right_on='user', suffixes=('','_friends'))
    
    merged_df2['is_creator_friend'] = merged_df2.apply (lambda row: is_event_creator_friend(row), axis=1)
    merged_df2 = merged_df2.drop(columns=['friends'])
    
    merged_df2['not_start_diff'] = merged_df2.apply (lambda row: time_to_event(row), axis=1)
    return merged_df2
    
def is_same_city(row):
    if (type(row['location']) != str) or (type(row['city']) != str):
        return False
    
    if row['location'].lower() in 'na' or row['city'].lower() in 'na':
        return False
    
    if str(row['city']).lower() in row['location'].lower():
        return True
    else:
        return False

def is_same_country(row):
    if (type(row['location']) != str) or (type(row['country']) != str):
        return False
    
    if row['location'].lower() in 'na' or row['country'].lower() in 'na':
        return False
        
    if row['country'].lower() in row['location'].lower():
        return True
    else:
        return False
    
    return row

    
def add_location_features(train_df, events_df, users_df):
    events_df = events_df[['event_id', 'city', 'country']]
    merged_df = pd.merge(train_df, events_df, how='inner', left_on='event', right_on='event_id', suffixes=('', '_event'))
    
    merged_df2 = pd.merge(merged_df, users_df, how='inner', left_on='user', right_on='user_id', suffixes=('','_user'))
    
    merged_df2['same_city'] =  merged_df2.apply (lambda row: is_same_city(row), axis=1)
    merged_df2['same_country'] = merged_df2.apply (lambda row: is_same_country(row), axis=1)
    merged_df2 = merged_df2[['invited', 'timestamp', 'interested', 'not_interested', 'event', 'same_city',
                            'same_country', 'user']]
    return merged_df2

def add_gender_age(train_df, users_df):
    users_df = users_df[['user_id', 'birthyear', 'gender']]
    merged_df = pd.merge(train_df, users_df, how='inner', left_on='user', right_on='user_id')
    cur_year = dt.datetime.now().year
    merged_df = merged_df.rename(columns = {'birthyear': 'age'})
    merged_df['age'] = merged_df['age'].astype(str).replace('None', np.nan)
    merged_df['age'] = merged_df['age'].ffill().bfill()
    merged_df['age'] = cur_year - pd.to_numeric(merged_df['age'])
    merged_df['gender'] = merged_df['gender'].ffill().bfill()
    return merged_df

def get_event_attendee_nums(event_attendees):
    event_attendee_nums_df = event_attendees[['event']]
    event_attendee_nums_df = pd.concat([event_attendee_nums_df, 
                                        event_attendees['yes'].str.len(),
                                        event_attendees['no'].str.len(),
                                        event_attendees['maybe'].str.len(),
                                        event_attendees['invited'].str.len()], axis=1)
    # print(event_attendee_nums_df.head(20))
    event_attendee_nums_df = event_attendee_nums_df.fillna(0)
    return event_attendee_nums_df
    # print(event_attendees['yes'].str.len())