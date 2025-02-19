import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping # type: ignore


# Load the datasets
file_path_energy = 'data/energy_dataset.csv'  # Adjusted file path
file_path_weather = 'data/weather_features.csv'  # Adjusted file path

energy_data = pd.read_csv(file_path_energy)
weather_data = pd.read_csv(file_path_weather)


numeric_columns = energy_data.select_dtypes(include=[np.number])


#Convert 'Time' to 'Datetime'
energy_data['time'] = pd.to_datetime(energy_data['time'], utc=True)
energy_data.set_index('time', inplace=True)


# Remove Columns
columns_to_remove_energy = ['generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead']

cleaned_energy_data = energy_data.drop(columns=columns_to_remove_energy)

#Remove Missing Values from Rows

cleaned_energy_data = energy_data.dropna()



# Convert the 'time' column to datetime and set it as the index
weather_data['dt_iso'] = pd.to_datetime(weather_data['dt_iso'], utc=True)
weather_data.set_index('dt_iso', inplace=True)

weather_data.info()

# Extract datetime features
#weather_data['dt_iso'] = pd.to_datetime(weather_data['dt_iso'])
#weather_data['year'] = weather_data['dt_iso'].dt.year
#weather_data['month'] = weather_data['dt_iso'].dt.month
#weather_data['day'] = weather_data['dt_iso'].dt.day
#weather_data['hour'] = weather_data['dt_iso'].dt.hour

# Remove irrelevant columns (e.g., 'weather_icon', 'weather_description')
irrelevant_columns = ['weather_icon', 'weather_description']
weather_data_cleaned = weather_data.drop(columns=irrelevant_columns)
print("\nWeather Dataset after removing irrelevant columns:")
print(weather_data_cleaned.head())

# Check for duplicate rows
duplicates = weather_data_cleaned.duplicated().sum()
print(f"\nNumber of duplicate rows in Weather Dataset: {duplicates}")
weather_data_cleaned = weather_data_cleaned.drop_duplicates()

print(f"\nNumber of duplicate rows in Weather Dataset: {weather_data_cleaned.duplicated().sum()}")

#3. Further Data Preparation and Merging of Dataaset

# Merge the datasets on the time index
combined_data = energy_data.merge(weather_data, how='outer', left_index=True, right_index=True)
combined_data.dropna(subset=['total load actual'], inplace=True)
combined_data['city_name'].fillna('Unknown', inplace=True)
combined_data['city_name'] = combined_data['city_name'].str.strip().str.lower()

combined_data.info()

# One-hot encode 'city_name'
city_encoded = pd.get_dummies(combined_data['city_name'], drop_first=False)
combined_data = pd.concat([combined_data[['total load actual']], city_encoded], axis=1)

# Feature scaling
scaler_load = MinMaxScaler()
scaler_city = MinMaxScaler()
scaled_load = scaler_load.fit_transform(combined_data[['total load actual']])
scaled_city = scaler_city.fit_transform(combined_data[city_encoded.columns])
scaled_data = np.hstack((scaled_load, scaled_city))

# Create sequences for Bi-LSTM
sequence_length = 24
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])


X, y = np.array(X), np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Bi-LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50, return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Train the model and capture the history
history = model.fit(
    X_train,
    y_train,
    epochs=10,  # Consider increasing epochs for better convergence
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)



# Function to predict future or past load
def predict_load(input_date, city_name, predict_past=False):
    input_date = pd.to_datetime(input_date, utc=True)

    # Adjust for past predictions
    if predict_past:
        if input_date >= energy_data.index[0]:
            raise ValueError("Input date must be before the dataset start date.")
    else:
        if input_date <= energy_data.index[-1]:
            raise ValueError("Input date must be in the future.")

    city_name = city_name.strip().lower()
    if city_name not in city_encoded.columns:
        raise ValueError(f"City name not found. Available cities: {', '.join(city_encoded.columns)}")

    future_sequence = scaled_data[:sequence_length, :].copy() if predict_past else scaled_data[-sequence_length:, :].copy()
    city_feature_index = list(city_encoded.columns).index(city_name)
    city_feature_values = np.zeros(len(city_encoded.columns))
    city_feature_values[city_feature_index] = 1
    city_feature_values_scaled = scaler_city.transform([city_feature_values])

    days_to_predict = abs((input_date - energy_data.index[0 if predict_past else -1]).days)

    for _ in range(days_to_predict):
        predicted_scaled = model.predict(future_sequence.reshape(1, sequence_length, -1))
        new_step = np.hstack((predicted_scaled, city_feature_values_scaled))
        future_sequence = np.vstack((new_step, future_sequence[:-1])) if predict_past else np.vstack((future_sequence[1:], new_step))

    predicted_load = scaler_load.inverse_transform(predicted_scaled.reshape(-1, 1))
    return predicted_load[0, 0]

# User input for prediction
user_input_date = input("Enter a date (YYYY-MM-DD): ")
user_input_city = input("Enter city name (e.g., Valencia): ")
predict_past = input("Predict for past? (yes/no): ").strip().lower() == 'yes'

try:
    predicted_load = predict_load(user_input_date, user_input_city, predict_past)
    print(f"Predicted total actual load for {user_input_date} in {user_input_city}: {predicted_load}")
except ValueError as e:
    print(e)

# Define a function for applying demand response adjustments
def apply_demand_response(predicted_load, time_index, strategy="peak_reduction", reduction_percent=20):
    """
    Adjusts the predicted load based on demand response strategies.

    Parameters:
    - predicted_load: Array of predicted load values.
    - time_index: Corresponding time indices for the predictions.
    - strategy: Type of demand response strategy. Options: 'peak_reduction', 'time_of_use'.
    - reduction_percent: Percentage reduction for peak load adjustment.

    Returns:
    - adjusted_load: Array of load values after applying demand response.
    """
    adjusted_load = predicted_load.copy()

    if strategy == "peak_reduction":
        # Identify peak hours (e.g., 6 PM to 10 PM)
        peak_hours = time_index[(time_index.hour >= 18) & (time_index.hour <= 22)]
        for peak_time in peak_hours:
            peak_indices = time_index == peak_time
            adjusted_load[peak_indices] *= (1 - reduction_percent / 100)

    elif strategy == "time_of_use":
        # Simulate time-of-use pricing adjustment
        off_peak_hours = time_index[(time_index.hour < 6) | (time_index.hour >= 22)]
        peak_hours = time_index[(time_index.hour >= 18) & (time_index.hour < 22)]

        # Apply reductions/increases based on time-of-use pricing
        for off_peak_time in off_peak_hours:
            off_peak_indices = time_index == off_peak_time
            adjusted_load[off_peak_indices] *= 0.8  # 20% reduction during off-peak
        for peak_time in peak_hours:
            peak_indices = time_index == peak_time
            adjusted_load[peak_indices] *= 1.1  # 10% increase during peak

    return adjusted_load
# Evaluate and plot predictions
y_pred = model.predict(X_test)
y_pred_rescaled = scaler_load.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = scaler_load.inverse_transform(y_test.reshape(-1, 1))
# Apply demand response adjustments to predictions
time_index = pd.date_range(start=energy_data.index[-len(y_test_rescaled):].min(),
                           periods=len(y_test_rescaled), freq='H')  # Example hourly index

adjusted_load = apply_demand_response(y_pred_rescaled, time_index, strategy="peak_reduction", reduction_percent=20)


# Include Demand Response in Predict Function
def predict_load_with_dr(input_date, city_name, strategy="peak_reduction", reduction_percent=20, predict_past=False):
    """
    Predicts load and applies demand response adjustments.

    Parameters:
    - input_date: Date for prediction (str, 'YYYY-MM-DD').
    - city_name: City name for prediction.
    - strategy: Demand response strategy. Options: 'peak_reduction', 'time_of_use'.
    - reduction_percent: Percentage reduction for peak load adjustment.
    - predict_past: Boolean for past or future prediction.

    Returns:
    - adjusted_predicted_load: Predicted load after demand response adjustment.
    """
    predicted_load = predict_load(input_date, city_name, predict_past)

    # Create a time index for simulation purposes
    time_index = pd.date_range(start=input_date, periods=24, freq='H')
    adjusted_predicted_load = apply_demand_response(
        np.array([predicted_load] * 24), time_index, strategy, reduction_percent
    )
    return adjusted_predicted_load

# Example: Predict future load with demand response
user_input_date = input("Enter a date for demand response prediction (YYYY-MM-DD): ")
user_input_city = input("Enter city name for demand response prediction: ")
strategy = input("Enter demand response strategy ('peak_reduction' or 'time_of_use'): ").strip()
reduction_percent = int(input("Enter percentage reduction for demand response: "))

try:
    adjusted_predicted_load = predict_load_with_dr(user_input_date, user_input_city, strategy, reduction_percent)
    print(f"Predicted load with demand response adjustment for {user_input_date}: {adjusted_predicted_load}")
except ValueError as e:
    print(e)



# Define the calculate_price function
def calculate_price(load_array, base_price=10, price_coefficient=0.001):
    """
    Calculates electricity price based on the predicted or adjusted load.

    Parameters:
    - load_array: Array of load values (e.g., predicted or adjusted load).
    - base_price: The minimum base price of electricity (e.g., €10/MWh).
    - price_coefficient: The rate at which price increases with load (e.g., €0.001 per MW).

    Returns:
    - price_array: Array of price values corresponding to the input load.
    """
    price_array = base_price + price_coefficient * np.array(load_array)
    return price_array


# Example: Predict future load with demand response
user_input_date = input("Enter a date for demand response prediction (YYYY-MM-DD): ")
user_input_city = input("Enter city name for demand response prediction: ")
strategy = input("Enter demand response strategy ('peak_reduction' or 'time_of_use'): ").strip()
reduction_percent = int(input("Enter percentage reduction for demand response: "))

try:
    # Compute the adjusted load using demand response
    adjusted_predicted_load = predict_load_with_dr(user_input_date, user_input_city, strategy, reduction_percent)

    # Check that adjusted_predicted_load is not empty
    if adjusted_predicted_load is None or len(adjusted_predicted_load) == 0:
        raise ValueError("Adjusted predicted load is empty. Please verify the prediction function.")

    # Calculate corresponding prices
    predicted_prices = calculate_price(adjusted_predicted_load)

    # Display results
    print(f"Predicted load with demand response adjustment for {user_input_date}: {adjusted_predicted_load}")
    print(f"Predicted prices corresponding to the adjusted load: {predicted_prices}")

    
except ValueError as e:
    print(e)

# Save the trained Bi-LSTM model
model.save("bi_lstm_model.keras")

# Save the MinMaxScaler used for preprocessing
import joblib
joblib.dump(scaler_load, "scaler.pkl")  # Saves the scaler for later use in Flask
