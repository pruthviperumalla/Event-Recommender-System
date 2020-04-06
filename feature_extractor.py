import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import datetime as dt

def add_location_features(train_df, events_df, users_df):
    events_df = events_df[['event_id', 'city', 'country']]
    merged_df = pd.merge(train_df, events_df, how='inner', left_on='event', right_on='event_id', suffixes=('', '_event'))
    return merged_df

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