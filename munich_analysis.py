# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:59:14 2020

@author: marku
"""
import os
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import seaborn as sns
import plotly.express as px
px.set_mapbox_access_token('pk.eyJ1IjoibWFya3VzZ3J1ZW5laXMiLCJhIjoiY2s5cjB2ODBhMDRkejNmb2JzcHBsY2ppNiJ9.kyBbd0mmZlxVPD3ru0X0Lw')
from sklearn.impute import SimpleImputer


#%%

data_path = r"C:\Users\marku\Airbnb\data"

df_calendar = pd.read_csv(os.path.join(data_path, "calendar.csv"))
df_listings = pd.read_csv(os.path.join(data_path, "listings.csv"))
#df_reviews = pd.read_csv(os.path.join(data_path, "reviews.csv"))


#%%
#Listings Dataset

df_listings_clean = df_listings.copy()

#%%
df_listings_clean = df_listings_clean.drop(['neighbourhood_group'], axis=1)


#%%
# rename id to listings id for further data handling
#TODO

#%%

#df_listings_clean['price_mean'] = df_listings_clean.groupby(['neighbourhood'])['price'].transform(lambda x : x.mean())

#%%
#Calendar Dataset
df_calendar_clean = df_calendar.copy()

#%%
def clean_price(df, cols, currency_column = False):
    '''
    function to convert price columns of string values into float values and a currency cloumn
    
    ARGS: 
    df: (data set of pandas dataframe type), 
    cols: list of dataframe column names that will be converted
    currency_column: boolean that indicates if a currency column already exists
    
    OUTPUT: 
    df: dataframe with converted price columns and currency column
    '''
    for col in cols:
        
        if currency_column:
            df[col] = df[col].str[1:]
            df[col] = df[col].replace({',':''}, regex=True).astype(float)
            
        else:
            df['currency'] = df[col].astype(str).str[0]
            currency_column = True
            df[col] = df[col].str[1:]
            df[col] = df[col].replace({',':''}, regex=True).astype(float)
        
    return df

#%%
 # date
df_calendar_clean['date'] = pd.to_datetime(df_calendar_clean['date'])   

#%%
#price
cols = ['price', 'adjusted_price']

df_calendar_clean = clean_price(df_calendar_clean, cols)


#%%
df_calendar_clean['price_diff'] = df_calendar_clean['price']-df_calendar_clean['adjusted_price']


#%%
df_calendar_clean_price = df_calendar_clean.groupby('date')['price'].mean().reset_index()


#%%
# Update: The Octoberfest 2020 is canceled due to Corona-Virus
# Originally the Oktoberfest was planned from September, 19th to October, 4th 2020
oktoberfest_dates = ["2020-09-19", "2020-10-04"]
# based on Google Maps
oktoberfest_coordinates = [48.131475, 11.549708]

#%%
df_calendar_clean["oktoberfest"] = (df_calendar_clean['date'] >= oktoberfest_dates[0]) & (df_calendar_clean['date'] <= oktoberfest_dates[1])
#%%

df_calendar_oktoberfest = df_calendar_clean[df_calendar_clean["oktoberfest"]==True]
df_calendar_no_oktoberfest = df_calendar_clean[df_calendar_clean["oktoberfest"]==False]


#%%
# listing oktober
df_calendar_clean_oktoberfest = df_calendar_oktoberfest.groupby('listing_id')['price'].mean().reset_index()


#%%
#listing no oktober
df_calendar_clean_no_oktoberfest = df_calendar_no_oktoberfest.groupby('listing_id')['price'].mean().reset_index()

#%%
#room

df_calendar_clean_oktoberfest = df_calendar_oktoberfest.groupby('listing_id')['price'].mean().reset_index()


#%%
df_calendar_clean_no_oktoberfest = df_calendar_no_oktoberfest.groupby('listing_id')['price'].mean().reset_index()

## Then show charts over all listings... siehe jupyter


#%%
# neu: connect with listings dataset

df_calendar_clean_oktoberfest.rename(columns = {'price':'oktoberfest_mean_price'}, inplace = True)
df_calendar_clean_no_oktoberfest.rename(columns = {'price':'no_oktoberfest_mean_price'}, inplace = True)


#%%
listings = df_listings_clean.copy()
#%%
oktoberfest = df_calendar_clean_oktoberfest.copy()
no_oktoberfest = df_calendar_clean_no_oktoberfest.copy()

#%%
#rename listings id siehe oben!

listings.rename(columns = {'id':'listing_id'}, inplace = True)

#%%


listings_new = pd.merge(listings, oktoberfest, on="listing_id")

#%%

listings_new = pd.merge(listings_new, no_oktoberfest, on="listing_id")



#%%
# calculate price difference percentage per listing

listings_new['Oktoberfest_effect'] = ((listings_new['oktoberfest_mean_price'] - listings_new['no_oktoberfest_mean_price']) / listings_new['no_oktoberfest_mean_price'] *100 )



#%%

lisitngs_new_type = listings_new.groupby('room_type').mean()

#%%

lisitngs_new_neighbourhood = listings_new.groupby('neighbourhood').mean()


#%%
# distance feature calculation based on latitude and longitude
#%%
def haversine(lat1, lon1, lat2, lon2, to_radians=False, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


#%%
#https://towardsdatascience.com/heres-how-to-calculate-distance-between-2-geolocations-in-python-93ecab5bbba4    

def haversine_distance(lat1, lon1, lat2, lon2):
   r = 6371
   phi1 = np.radians(lat1)
   phi2 = np.radians(lat2)
   delta_phi = np.radians(lat2 - lat1)
   delta_lambda = np.radians(lon2 - lon1)
   a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
   res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
   return np.round(res, 2)    


#%%
   
#oktoberfest_la

#%%
   
distances_km = []
for row in listings_new.itertuples(index=False):
   distances_km.append(
       haversine_distance(oktoberfest_coordinates[0], oktoberfest_coordinates[1], row.latitude, row.longitude)
   )
   
   
#%%

listings_new['Distance'] =distances_km   
   
   
#%%

plt.scatter(listings_new['Distance'], listings_new['Oktoberfest_effect'])
plt.show()

#%%














