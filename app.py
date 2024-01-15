import numpy as np
import flask
import io
import datetime
import tensorflow as tf
from constants import LATEST_MODEL_PATH,bucharest_station,NORM_DATA_PATH,past_days
import pickle
from meteostat import Hourly
import pandas as pd

# initialize our Flask application and the Keras model
model = None
norm_data = None


def load_model():
    global model
    global norm_data
    model = tf.keras.models.load_model(LATEST_MODEL_PATH)
    with open(NORM_DATA_PATH,"rb") as f:
        norm_data = pickle.load(f)
    print(norm_data)

def get_temp():

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=past_days)
    weather_data = Hourly(bucharest_station,start_date,end_date).fetch()
    weather_data = weather_data[['temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres']]
    weather_data = weather_data.dropna(axis=1,how="any")
    hours_sin = np.sin(weather_data.index.map(pd.Timestamp.timestamp).values / (24*60*60)*2*np.pi)
    days_sin = np.sin(weather_data.index.map(pd.Timestamp.timestamp).values / (365.2425*24*60*60)*2*np.pi)
    hours_cos = np.cos(weather_data.index.map(pd.Timestamp.timestamp).values / (24*60*60)*2*np.pi)
    days_cos = np.cos(weather_data.index.map(pd.Timestamp.timestamp).values / (365.2425*24*60*60)*2*np.pi)

    data = (weather_data.values - norm_data[1] ) / norm_data[0]
    data = pd.concat([pd.DataFrame(data),
                      pd.DataFrame(hours_sin,columns=[len(list(weather_data.columns))]),
                      pd.DataFrame(days_sin,columns=[len(list(weather_data.columns))+1]),
                      pd.DataFrame(hours_cos,columns=[len(list(weather_data.columns))+2]),
                      pd.DataFrame(days_cos,columns=[len(list(weather_data.columns))+3])
                      ],axis=1).values
    data = np.array([data])
    out = model.predict(data)[0]
    out = out * norm_data[2] + norm_data[3]
    temp = out[0]
    # humidity = out[1]
    return round(temp), 0#round(humidity)

load_model()
app = flask.Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    data = {"success": False}

    try:
        t,h = get_temp()
        data["temp"] = t
        # data["humidity"] = h
        data["success"] = True
    except Exception as e:
        data["reason"] = str(e)
        return flask.jsonify(data)
    return flask.jsonify(data)

@app.route("/",methods=["GET"])
def page():
    t,h = get_temp()
    return flask.render_template("landing_page.html",temp=f"{t} °C",hum=f"{h}%")

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    
    app.run()