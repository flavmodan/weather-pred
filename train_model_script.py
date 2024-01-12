from meteostat import Hourly
from datetime import timedelta,datetime
from constants import bucharest_station,LATEST_MODEL_PATH,NORM_DATA_PATH,past,future
import pickle
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

start_date = datetime(2021,1,15)
end_date = datetime(2024,1,11)

weather_data = Hourly(bucharest_station,start_date,end_date).fetch()
weather_data = weather_data[['temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres']]
weather_data = weather_data.dropna(axis=1,how="any")
hours_sin = np.sin(weather_data.index.map(pd.Timestamp.timestamp).values / (24*60*60)*2*np.pi)
days_sin = np.sin(weather_data.index.map(pd.Timestamp.timestamp).values / (365.2425*24*60*60)*2*np.pi)
hours_cos = np.cos(weather_data.index.map(pd.Timestamp.timestamp).values / (24*60*60)*2*np.pi)
days_cos = np.cos(weather_data.index.map(pd.Timestamp.timestamp).values / (365.2425*24*60*60)*2*np.pi)
# hours = np.sin(hours*2*np.pi)
num_of_features = len(list(weather_data.columns)) + 4

split_fraction = 0.715
train_split = int(split_fraction * int(weather_data.shape[0]))
step = 1

learning_rate = 0.001
batch_size = 256
epochs = 10

def normalize(series,train_split):
    mean = series[:train_split].mean(axis=0)
    std = series[:train_split].std(axis=0)
    with open(NORM_DATA_PATH,"wb") as f:
        pickle.dump((std,mean,std[0],mean[0]),f)
    return (series - mean) / std

features = normalize(weather_data.values, train_split)
features = pd.concat([pd.DataFrame(features),
                      pd.DataFrame(hours_sin,columns=[len(list(weather_data.columns))]),
                      pd.DataFrame(days_sin,columns=[len(list(weather_data.columns))+1]),
                      pd.DataFrame(hours_cos,columns=[len(list(weather_data.columns))+2]),
                      pd.DataFrame(days_cos,columns=[len(list(weather_data.columns))+3])
                      ],axis=1)

train_data = features.loc[0 : train_split - 1]
val_data = features.loc[train_split:]

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(num_of_features)]].values
y_train = features.iloc[start:end][[1]]

sequence_length = int(past / step)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(num_of_features)]].values
y_val = features.iloc[label_start:][[1]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

path_checkpoint = LATEST_MODEL_PATH
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_best_only=True,
    mode="max"
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

visualize_loss(history, "Training and Validation Loss")


def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return


for x, y in dataset_val.take(5):
    show_plot(
        [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
        12,
        "Single Step Prediction",
    )

# points_before = 7*24
# points_after = 6

# batch_size = 64

# # hours = weather_data.index.hour
# # weather_data = pd.concat([weather_data,pd.DataFrame(hours,index=weather_data.index)],axis=1)
# temperature = weather_data["temp"]
# temp_mean = temperature.mean(axis=0)
# temp_std = temperature.std(axis=0)

# def normalize(series):
#     mean = series.mean(axis=0)
#     std = series.std(axis=0)
#     with open(NORM_DATA_PATH,"wb") as f:
#         pickle.dump((std,mean,temp_std,temp_mean),f)
#     return (series - mean) / std



# features = normalize(weather_data.values)
# features = pd.DataFrame(features)

# train_split = 0.8

# train_size = int(train_split * features.shape[0])

# train_data = features[0:train_size+1]
# val_data = features.loc[train_size:]

# start = points_before+points_after # hours in the past to get as input + hours in the future to predict
# end = start + train_size

# x_train = train_data.values
# y_train = features.iloc[start:end]

# sequence_length = int(points_before)

# from tensorflow import keras

# dataset_train = keras.utils.timeseries_dataset_from_array(
#     data=x_train,
#     targets=y_train,
#     sequence_length=sequence_length,
#     batch_size=batch_size,
# )

# x_val_end = len(val_data) - start

# label_start = train_size + start

# x_val = val_data.iloc[:x_val_end][[i for i in range(num_of_features)]].values
# y_val = features.iloc[label_start:][[0]]

# dataset_val = keras.utils.timeseries_dataset_from_array(
#     x_val,
#     y_val,
#     sequence_length=sequence_length,
#     batch_size=batch_size,
# )

# for batch in dataset_train.take(1):
#     inputs, targets = batch
    
# inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
# d = keras.layers.LSTM(32)(inputs)
# # d = keras.layers.Dense(32,activation="relu")(inputs)
# # d = keras.layers.Dense(32)(d)
# outputs = keras.layers.Dense(1)(d)#(lstm_out)
# model = keras.Model(name="Weather_forcaster",inputs=inputs, outputs=outputs)




# # model.compile(optimizer=keras.optimizers.Nadam(
# #     learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
# #     name='Nadam'
# # ), loss="mse")

# model.compile(optimizer="adam",loss="mse")

# model.summary()

# history = model.fit(
#     dataset_train,
#     epochs=15,
#     validation_data=dataset_val
# )

# model.save(LATEST_MODEL_PATH)

# import matplotlib.pyplot as plt


# for x, y in dataset_val:
#     history_data = x[0][:, 1].numpy() * temp_std + temp_mean
#     true_value = y[0].numpy() * temp_std + temp_mean
#     prediction = model.predict(x)[0] * temp_std + temp_mean
#     time_steps = list(range(-(history_data.shape[0]), 0))
#     plt.plot(time_steps, history_data)
#     plt.plot(points_after, true_value, "gD")
#     plt.plot(points_after, prediction, "rX")
#     plt.legend(["History", "True Future", "Model Prediction"])
#     plt.xlabel("Time")
#     plt.show()