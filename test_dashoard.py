import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the datasets directly from CSV
file_path_energy = 'data/energy_dataset.csv'
file_path_weather = 'data/weather_features.csv'

energy_data = pd.read_csv(file_path_energy, parse_dates=['time'])
energy_data['time'] = pd.to_datetime(energy_data['time'], utc=True)
energy_data.set_index("time", inplace=True)

weather_data = pd.read_csv(file_path_weather, parse_dates=['dt_iso'])
weather_data['dt_iso'] = pd.to_datetime(weather_data['dt_iso'], utc=True)
weather_data.set_index("dt_iso", inplace=True)

# Merge datasets
combined_data = energy_data.merge(weather_data, how='outer', left_index=True, right_index=True)
combined_data.dropna(subset=['total load actual'], inplace=True)
combined_data['city_name'].fillna('Unknown', inplace=True)
combined_data['city_name'] = combined_data['city_name'].str.strip().str.lower()

# One-hot encode the city names
city_encoded = pd.get_dummies(combined_data['city_name'], drop_first=False)
combined_data = pd.concat([combined_data[['total load actual']], city_encoded], axis=1)

# Scale features
scaler_load = MinMaxScaler()
scaler_city = MinMaxScaler()
scaled_load = scaler_load.fit_transform(combined_data[['total load actual']])
scaled_city = scaler_city.fit_transform(combined_data[city_encoded.columns])

# Combine scaled features into one array
scaled_data = np.hstack((scaled_load, scaled_city))

# Define sequence length (used in creating sequences)
sequence_length = 24

# Initialize Flask app
app = Flask(__name__)

# Load trained Bi-LSTM model
model = tf.keras.models.load_model("bi_lstm_model.keras")

# Load Fault Detection Model and Scaler 
fault_model = joblib.load("rf_fault_detection_model.pkl")
# If your fault detection model was trained on scaled data, load its scaler.
fault_scaler = joblib.load("fault_scaler.pkl")


# (The following scaler saving/loading code is optional depending on your training workflow)
X_train = np.random.rand(100, 13)
scaler = MinMaxScaler()
scaler.fit(X_train)
joblib.dump(scaler, "feature_scaler.pkl")
y_train = np.random.rand(100, 1)
target_scaler = MinMaxScaler()
target_scaler.fit(y_train)
joblib.dump(target_scaler, "target_scaler.pkl")
scaler = joblib.load("scaler.pkl")
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

def predict_load(input_date, city, predict_past=False):
    input_date = pd.to_datetime(input_date, utc=True)
 
    city = city.strip().lower()
    city_mapping = {'valencia': 0, 'bilbao': 1, 'seville': 2, 'madrid': 3, 'barcelona': 4}
    if city not in city_mapping:
        raise ValueError(f"City name not found. Available cities: {', '.join(city_mapping.keys())}")
    
    # Create one-hot vector for city
    city_vector = np.zeros((1, len(city_mapping)), dtype=np.float32)
    city_vector[0, city_mapping[city]] = 1.0

    # Use the last sequence from your scaled data
    future_sequence = scaled_data[-sequence_length:, :].copy()
    days_to_predict = abs((input_date - energy_data.index[-1]).days)
    
    for _ in range(days_to_predict):
        predicted_scaled = model.predict(future_sequence.reshape(1, sequence_length, -1))
        new_step = np.hstack((predicted_scaled, city_vector))
        future_sequence = np.vstack((future_sequence[1:], new_step))
    
    predicted_load = scaler_load.inverse_transform(predicted_scaled.reshape(-1, 1))
    return float(predicted_load[0, 0])

def apply_demand_response(predicted_load, strategy="peak_reduction", reduction_percent=20):
    adjusted_load = predicted_load
    if strategy == "peak_reduction":
        adjusted_load *= (1 - reduction_percent / 100)
    elif strategy == "time_of_use":
        adjusted_load *= (0.9 if reduction_percent > 10 else 1.1)
    return adjusted_load

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

from datetime import datetime

@app.route("/load-forecast-page", methods=["GET", "POST"])
def load_forecast():
    current_date = datetime.utcnow().strftime('%Y-%m-%d')
    prediction = None
    error = None
    city = None
    if request.method == "POST":
        try:
            date = request.form.get("date")
            city = request.form.get("city")
            predicted_load = predict_load(date, city)
            prediction = (date, predicted_load)
        except Exception as e:
            error = str(e)
            
    
    return render_template("load-forecast-page.html", 
                           prediction=prediction,
                           error=error,
                           city=city,
                           current_date=current_date)

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

@app.route("/demand-response-page", methods=["GET", "POST"])
def demand_response_page():
    if request.method == "POST":
        date = request.form["date"]
        city = request.form["city"]
        strategy = request.form["strategy"]
        reduction_percent = int(request.form["reduction"])
        predicted_load = predict_load(date, city)
        adjusted_load = apply_demand_response(predicted_load, strategy, reduction_percent)
        predicted_price = calculate_price([predicted_load])[0]
        adjusted_price = calculate_price([adjusted_load])[0]
        prediction = (date, predicted_load, predicted_price, adjusted_load, adjusted_price)
        
        # Generate time-of-use data (example using a 24-hour forecast)
        predicted_date = pd.to_datetime(date)
        time_index = pd.date_range(start=predicted_date, periods=24, freq='H')
        generated_load_array = np.full(24, predicted_load)
        adjusted_load_array  = np.full(24, adjusted_load)
        predicted_price_array = np.full(24, predicted_price)
        
        df = pd.DataFrame({
            'Time': time_index,
            'Generated_Load': generated_load_array,
            'Adjusted_Load': adjusted_load_array,
            'Predicted_Price': predicted_price_array
        })
        
        df['Time_Period'] = pd.cut(
            df['Time'].dt.hour,
            bins=[-1, 11, 17, 24],
            labels=['Morning', 'Afternoon', 'Night'],
            right=False
        )
        
        averages = df.groupby('Time_Period')[['Generated_Load', 'Adjusted_Load', 'Predicted_Price']].mean().reset_index()
        time_periods_data = averages.to_dict(orient='records')
        
        return render_template("demand-response-page.html", 
                               city=city, 
                               prediction=prediction,
                               time_periods_data=time_periods_data)
    else:
        # Pass an empty list so that the template can handle it gracefully
        return render_template("demand-response-page.html", time_periods_data=[])

#  Fault Detection Prediction Function ---
def predict_fault(Ia, Ib, Ic, Va, Vb, Vc):
    """
    Predicts whether a fault is present based on electrical measurements.
    Parameters:
        Ia, Ib, Ic, Va, Vb, Vc: float values representing the currents and voltages.
    Returns:
        A string: "Fault" if a fault is detected, "No Fault" otherwise.
    """
    # Prepare the input array for prediction
    features = np.array([[Ia, Ib, Ic, Va, Vb, Vc]])
    # Scale features using the fault detection scaler (if used during training)
    features_scaled = fault_scaler.transform(features)
    prediction = fault_model.predict(features_scaled)
    return "Fault" if prediction[0] == 1 else "No Fault"

# --- Fault Detection API Endpoint ---
@app.route("/fault-detection-page", methods=["GET", "POST"])
def fault_detection():
    """
    API endpoint for fault detection.
    Expects a JSON payload with keys: 'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'.
    Returns a JSON with the fault status.
    """
    try:
        data = request.get_json(force=True)
        Ia = float(data.get("Ia"))
        Ib = float(data.get("Ib"))
        Ic = float(data.get("Ic"))
        Va = float(data.get("Va"))
        Vb = float(data.get("Vb"))
        Vc = float(data.get("Vc"))
        fault_status = predict_fault(Ia, Ib, Ic, Va, Vb, Vc)
        return jsonify({"fault_status": fault_status})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
