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
    # t_in = data[:days_before*24]
    data = np.array([data])
    out = model.predict(data)[0][0]

    # tt_out = data[days_before*24+hours_after][0] * norm_data[2] + norm_data[3]

    temp = out * norm_data[2] + norm_data[3]
    return round(temp)

load_model()
app = flask.Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    try:
        temp = get_temp()
        data["temp"] = temp
        data["success"] = True
    except Exception as e:
        data["reason"] = str(e)
        return flask.jsonify(data)
     # return the data dictionary as a JSON response
    return flask.jsonify(data)

@app.route("/",methods=["GET"])
def page():
    temp = get_temp()
    return flask.render_template("landing_page.html",temp=f"{temp} °C")

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    
    app.run()