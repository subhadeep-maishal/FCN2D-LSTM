import os
import xarray as xr
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to replace NaN values with the mean of each variable
def replace_nan_with_mean(data):
    nan_mean = np.nanmean(data)
    data[np.isnan(data)] = nan_mean
    return data

# Set environment variables for parallelism (32 cores)
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['TF_NUM_INTRAOP_THREADS'] = '32'
os.environ['TF_NUM_INTEROP_THREADS'] = '32'

# Limit TensorFlow to use 32 cores for parallel computation
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)

# Load the dataset
netcdf_file = r"/scratch/20cl91p02/ANN_BIO/FCN/ann_input_data.nc"
ds = xr.open_dataset(netcdf_file)

# Extract data variables
fe = ds['fe'].values  # (time, depth, lat, lon)
po4 = ds['po4'].values
si = ds['si'].values
no3 = ds['no3'].values  # Predictor
nppv = ds['nppv'].values  # Target variable

# Replace NaN values with the mean of each variable using the custom function
fe = replace_nan_with_mean(fe)
po4 = replace_nan_with_mean(po4)
si = replace_nan_with_mean(si)
no3 = replace_nan_with_mean(no3)
nppv = replace_nan_with_mean(nppv)

# Since depth is constant, discard the depth dimension and focus on (time, lat, lon)
fe = fe[:, 0, :, :]
po4 = po4[:, 0, :, :]
si = si[:, 0, :, :]
no3 = no3[:, 0, :, :]
nppv = nppv[:, 0, :, :]  # Ensure this matches the structure

# Stack input variables along a new channel dimension
inputs = np.stack([fe, po4, si, no3], axis=-1)  # Shape: (time, lat, lon, channels)

# Split data for training and testing
train_size = int(0.8 * inputs.shape[0])
X_train, X_test = inputs[:train_size], inputs[train_size:]
y_train, y_test = nppv[:train_size], nppv[train_size:]

# Prepare input for LSTM
time_steps = 5  # Number of time steps to consider in each sequence
train_samples = X_train.shape[0] - time_steps
test_samples = X_test.shape[0] - time_steps

X_train_lstm = np.array([X_train[i:i + time_steps] for i in range(train_samples)])
y_train_lstm = y_train[time_steps:]  # Shape: (samples, lat, lon)
X_test_lstm = np.array([X_test[i:i + time_steps] for i in range(test_samples)])
y_test_lstm = y_test[time_steps:]

# Normalize the data
scaler_X = StandardScaler()
X_train_lstm_reshaped = X_train_lstm.reshape(-1, X_train_lstm.shape[2] * X_train_lstm.shape[3] * X_train_lstm.shape[4])
X_train_lstm_scaled = scaler_X.fit_transform(X_train_lstm_reshaped).reshape(X_train_lstm.shape)

X_test_lstm_reshaped = X_test_lstm.reshape(-1, X_test_lstm.shape[2] * X_test_lstm.shape[3] * X_test_lstm.shape[4])
X_test_lstm_scaled = scaler_X.transform(X_test_lstm_reshaped).reshape(X_test_lstm.shape)

scaler_y = StandardScaler()
y_train_lstm_reshaped = y_train_lstm.reshape(-1, y_train_lstm.shape[1] * y_train_lstm.shape[2])
y_train_lstm_scaled = scaler_y.fit_transform(y_train_lstm_reshaped).reshape(y_train_lstm.shape)

y_test_lstm_reshaped = y_test_lstm.reshape(-1, y_test_lstm.shape[1] * y_test_lstm.shape[2])
y_test_lstm_scaled = scaler_y.transform(y_test_lstm_reshaped).reshape(y_test_lstm.shape)

# Define the FCN + LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                                     input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2], X_train_lstm.shape[3], X_train_lstm.shape[4])),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.LSTM(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train_lstm.shape[1] * y_train_lstm.shape[2])  # Flattened (lat * lon)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_lstm_scaled, y_train_lstm_scaled.reshape(y_train_lstm_scaled.shape[0], -1),
                    validation_data=(X_test_lstm_scaled, y_test_lstm_scaled.reshape(y_test_lstm_scaled.shape[0], -1)),
                    epochs=50, batch_size=16)

# Make predictions
predictions = model.predict(X_test_lstm_scaled)

# Compute averages
average_actual_nppv = np.nanmean(y_test, axis=0)  # Average across time dimension
average_predicted_nppv = np.nanmean(predictions, axis=0)

# Reshape `average_predicted_nppv` to match latitude and longitude dimensions
latitude_size = ds['latitude'].size
longitude_size = ds['longitude'].size
average_predicted_nppv = average_predicted_nppv.reshape((latitude_size, longitude_size))

# Save results to NetCDF
output_file_path = r"/scratch/20cl91p02/ANN_BIO/FCN/average_output_fcn+lstm_nppv_parallel.nc"
with xr.Dataset() as ds_out:
    ds_out.coords['latitude'] = ('latitude', ds['latitude'].values)
    ds_out.coords['longitude'] = ('longitude', ds['longitude'].values)
    ds_out['average_actual_nppv'] = (('latitude', 'longitude'), average_actual_nppv)
    ds_out['average_predicted_nppv'] = (('latitude', 'longitude'), average_predicted_nppv)
    ds_out.attrs['title'] = 'Average NPPV Concentrations (FCN + LSTM)'
    ds_out.to_netcdf(output_file_path)
    print(f"Output saved to: {output_file_path}")
