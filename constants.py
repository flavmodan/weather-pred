from meteostat import Stations

bucharest_coords = (44.42859452413485, 26.104589551084285)

weather_stations = Stations()

bucharest_station = weather_stations.nearby(lat=bucharest_coords[0],lon=bucharest_coords[1]).fetch(1)

LATEST_MODEL_PATH = f"models/model.keras"
NORM_DATA_PATH = "archive/norm_data.pkl"

past_days = 4
past = past_days*24
future = 2