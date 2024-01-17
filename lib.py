from meteostat import Stations
import pickle 
import pandas as pd
from meteostat import Hourly
import numpy as np

bucharest_coords = (44.42859452413485, 26.104589551084285)

weather_stations = Stations()

bucharest_station = weather_stations.nearby(lat=bucharest_coords[0],lon=bucharest_coords[1]).fetch(1)

LATEST_MODEL_PATH = f"models/model.keras"
NORM_DATA_PATH = "archive/norm_data.pkl"

past_days = 7
past = past_days*24
future = 2

features_to_predict = [1]

split_fraction = 0.8

step = 1

batch_size = 256
epochs = 20

def get_weather_data(start_date,end_date,for_train=False,norm_data=[]):
    
    weather_data = Hourly(bucharest_station,start_date,end_date).fetch()
    weather_data = weather_data[['temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres']]
    weather_data = weather_data.dropna(axis=1,how="any")
    train_split = int(split_fraction * int(weather_data.shape[0]))
    hours = weather_data.index.map(pd.Timestamp.timestamp).values / (24*60*60)*2*np.pi
    days = weather_data.index.map(pd.Timestamp.timestamp).values / (365.2425*24*60*60)*2*np.pi
    num_of_features = len(list(weather_data.columns)) + 2


    def normalize(series,train_split):
        mean = series[:train_split].mean(axis=0)
        std = series[:train_split].std(axis=0)
        with open(NORM_DATA_PATH,"wb") as f:
            pickle.dump((std,mean,std[np.array(features_to_predict) - 1],mean[np.array(features_to_predict) - 1]),f)
        return (series - mean) / std

    if for_train:
        features = normalize(weather_data.values, train_split)
    else:
        features = (weather_data.values - norm_data[1] ) / norm_data[0]

    features = pd.concat([pd.DataFrame(features),
                        pd.DataFrame(hours,columns=[len(list(weather_data.columns))]),
                        pd.DataFrame(days,columns=[len(list(weather_data.columns))+1]),
                        ],axis=1)
    
    return features,num_of_features,train_split