from copy import deepcopy
from datetime import datetime
from lib import *

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

start_date = datetime(2021,1,15)
end_date = datetime(2024,1,13)

# generate train and evaluation datasets
features,num_of_features,train_split = get_weather_data(start_date,end_date,for_train=True)

train_data = features.loc[0 : train_split - 1]
val_data = features.loc[train_split:]

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(num_of_features)]].values
y_train = features.iloc[start:end][features_to_predict]

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
y_val = features.iloc[label_start:][features_to_predict]

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

input_shape = inputs.numpy().shape
output_shape = targets.numpy().shape

# func to generate model for grid search

def generate_model(lstm_units,dense_configuration,dense_units_for_time_periods,learning_rate):
    inputs = keras.layers.Input(shape=(input_shape[1], input_shape[2]))
    w_tensor = inputs[:,:,:input_shape[2]-2]
    time_tensor = inputs[:,:,input_shape[2]-2:]
    time_out = keras.layers.Dense(dense_units_for_time_periods,activation="tanh")(time_tensor)
    lstm_in = keras.layers.concatenate([w_tensor,time_out])
    lstm_out = keras.layers.LSTM(lstm_units)(lstm_in)
    for num_of_units in dense_configuration:
        lstm_out = keras.layers.Dense(num_of_units)(lstm_out)
    h = keras.layers.Dense(len(features_to_predict))(lstm_out)
    model = keras.Model(inputs=inputs, outputs=h)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse",metrics=["cosine_similarity"])
    model.summary()
    return model

# evaluate model
def get_mode_performance(model):
    res = model.evaluate(dataset_val)
    return res

# compare performances
def is_better_perf(best_so_far,now):
    if best_so_far is None:
        return True
    if best_so_far[1] < now[1] and best_so_far[0] > now[0]:
        return True
    return False

# train model and return performance
def train_and_check(model):
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[],
    )

    return get_mode_performance(model),history

# get deepcopy of a model
def get_model_copy(model):
    # return tf.keras.models.clone_model(model)
    model.save("temp.keras")
    return tf.keras.models.load_model("temp.keras")

# parameters for grid search
max_number_of_dense_layers = 2
dense_counts_per_layer = [16,32]
dense_units_for_time = [4,8,12]
lstm_unit_counts = [32,64,128]
learning_rates = [0.001,0.0005,0.0001]

best_model = None
best_model_history = None
best_model_performance = None

num_of_models = len(dense_units_for_time)*len(lstm_unit_counts)*len(learning_rates)*(1+(max_number_of_dense_layers-1)*len(dense_counts_per_layer))
model_num = 1

# gridsearch best arhitecture

for num_of_dense in range(max_number_of_dense_layers):
    dense_counts_to_try = dense_counts_per_layer
    if num_of_dense == 0:
        dense_counts_to_try = [0]
    for dense_count in dense_counts_to_try:
        for time_dense in dense_units_for_time:
            for lstm_units in lstm_unit_counts:
                for lr in learning_rates:
                    print(f"\nModel {model_num} out of {num_of_models}\n")
                    model_num+=1
                    model = generate_model(lstm_units,[dense_count]*num_of_dense,time_dense,lr)
                    perf,hist = train_and_check(model)
                    print(f"Model performace {perf}")
                    if is_better_perf(best_model_performance,perf):
                        best_model = get_model_copy(model)
                        best_model_performance = deepcopy(perf)
                        best_model_history = hist
                        print(f"Updated best model performance to {best_model_performance}")

# show and save best model 
model = get_model_copy(best_model)
print(f"best model summary :")
model.summary()
perf = get_mode_performance(model)
print(f"best model perf {perf}")
model.save(LATEST_MODEL_PATH)

# visualize best model performance on real unseen data
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

visualize_loss(best_model_history, "Training and Validation Loss")


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
        [x[0][:, 1].numpy(), y[0].numpy()[0], model.predict(x)[0][0]],
        future,
        "Single Step Prediction",
    )