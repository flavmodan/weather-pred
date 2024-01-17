import numpy as np
import flask
import io
import datetime
import tensorflow as tf
from lib import *
import pickle
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
    data,_,__ = get_weather_data(start_date,end_date,norm_data=norm_data)
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