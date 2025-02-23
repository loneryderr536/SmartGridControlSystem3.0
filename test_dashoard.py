import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load datasets
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

# One-hot encode city names
city_encoded = pd.get_dummies(combined_data['city_name'], drop_first=False)
combined_data = pd.concat([combined_data[['total load actual']], city_encoded], axis=1)

# Scale features
scaler_load = MinMaxScaler()
scaler_city = MinMaxScaler()
scaled_load = scaler_load.fit_transform(combined_data[['total load actual']])
scaled_city = scaler_city.fit_transform(combined_data[city_encoded.columns])

# Combine scaled features
scaled_data = np.hstack((scaled_load, scaled_city))

# Define sequence length
sequence_length = 24

# Load trained Bi-LSTM model
model = tf.keras.models.load_model("bi_lstm_model.keras")

# Load Fault Detection Model and Scaler
fault_model = joblib.load("rf_fault_detection_model.pkl")
fault_scaler = joblib.load("fault_scaler.pkl")

# City mapping
city_mapping = {'valencia': 0, 'bilbao': 1, 'seville': 2, 'madrid': 3, 'barcelona': 4}

# ----------------- Load Forecasting -----------------
def predict_load(input_date, city, predict_past=False):
    """
    Predicts the electricity load for a given city and date.
    Returns a 24-hour forecast.
    """
    input_date = pd.to_datetime(input_date, utc=True)
    city = city.strip().lower()

    if city not in city_mapping:
        raise ValueError(f"City name not found. Available cities: {', '.join(city_mapping.keys())}")
    
    # Create one-hot city vector
    city_vector = np.zeros((1, len(city_mapping)), dtype=np.float32)
    city_vector[0, city_mapping[city]] = 1.0

    # Use the last sequence for prediction
    future_sequence = scaled_data[-sequence_length:, :].copy()
    
    predicted_loads = []
    for _ in range(24):  # Predict for 24 hours
        predicted_scaled = model.predict(future_sequence.reshape(1, sequence_length, -1))
        new_step = np.hstack((predicted_scaled, city_vector))
        future_sequence = np.vstack((future_sequence[1:], new_step))
        predicted_load = scaler_load.inverse_transform(predicted_scaled.reshape(-1, 1))[0, 0]
        predicted_loads.append(predicted_load)

    return np.array(predicted_loads)

# ----------------- Demand Response Strategies -----------------
def apply_peak_reduction(predicted_load, time_index, reduction_percent=20):
    """
    Reduces load during peak hours (6 PM - 10 PM).
    """
    adjusted_load = np.array(predicted_load)
    peak_hours = (time_index.hour >= 18) & (time_index.hour <= 22)
    adjusted_load[peak_hours] *= (1 - reduction_percent / 100)
    return adjusted_load

def apply_time_of_use(predicted_load, time_index):
    """
    Applies Time-of-Use (ToU) pricing strategy.
    """
    adjusted_load = np.array(predicted_load)
    off_peak_hours = (time_index.hour < 6) | (time_index.hour >= 22)  # Midnight - 6 AM, after 10 PM
    peak_hours = (time_index.hour >= 18) & (time_index.hour < 22)  # 6 PM - 10 PM

    adjusted_load[off_peak_hours] *= 0.8  # 20% reduction during off-peak
    adjusted_load[peak_hours] *= 1.1  # 10% increase during peak
    return adjusted_load

# ----------------- Price Calculation -----------------
def calculate_price(load_array, base_price=10, price_coefficient=0.001):
    """
    Calculates electricity price based on predicted or adjusted load.
    """
    return base_price + price_coefficient * np.array(load_array)

# ----------------- Flask Routes -----------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/load-forecast-page", methods=["GET", "POST"])
def load_forecast():
    current_date = datetime.utcnow().strftime('%Y-%m-%d')
    prediction = None
    error = None

    if request.method == "POST":
        try:
            date = request.form.get("date")
            city = request.form.get("city")
            predicted_load = predict_load(date, city)
            prediction = (date, np.mean(predicted_load))
        except Exception as e:
            error = str(e)

    return render_template("load-forecast-page.html", 
                           prediction=prediction,
                           error=error,
                           current_date=current_date)

@app.route("/demand-response-page", methods=["GET", "POST"])
def demand_response_page():
    try:
        if request.method == "POST":
            date = request.form.get("date")
            city = request.form.get("city")
            strategy = request.form.get("strategy")
            reduction_percent = int(request.form.get("reduction", 20))

            if not date or not city:
                raise ValueError("Date and City are required.")

            # Generate prediction
            predicted_date = pd.to_datetime(date)
            time_index = pd.date_range(start=predicted_date, periods=24, freq='H')
            generated_load_array = predict_load(date, city)

            # Apply correct strategy
            if strategy == "peak_reduction":
                adjusted_load_array = apply_peak_reduction(generated_load_array, time_index, reduction_percent)
            elif strategy == "time_of_use":
                adjusted_load_array = apply_time_of_use(generated_load_array, time_index)  # ✅ FIXED

            else:
                raise ValueError("Invalid strategy selected.")

            # Calculate prices
            predicted_price_array = calculate_price(generated_load_array)
            adjusted_price_array = calculate_price(adjusted_load_array)

            # Create DataFrame
            df = pd.DataFrame({
                'Time': time_index.strftime('%Y-%m-%d %H:%M:%S'),
                'Generated_Load': generated_load_array,
                'Adjusted_Load': adjusted_load_array,
                'Predicted_Price': predicted_price_array,
                'Adjusted_Price': adjusted_price_array
            })
            df['Time'] = pd.to_datetime(df['Time'])
            df['Time_Period'] = pd.cut(df['Time'].dt.hour, bins=[-1, 11, 17, 24], labels=['Morning', 'Afternoon', 'Night'], right=False)

            # Aggregate
            grouped_df = df.groupby('Time_Period')[['Generated_Load', 'Adjusted_Load', 'Predicted_Price']].mean().reset_index()
            time_periods_data = grouped_df.to_dict(orient='records')

            return render_template("demand-response-page.html", 
                                   city=city, 
                                   prediction=(date, np.mean(generated_load_array), np.mean(predicted_price_array),
                                               np.mean(adjusted_load_array), np.mean(adjusted_price_array)),
                                   time_periods_data=time_periods_data)

        else:
            return render_template("demand-response-page.html", time_periods_data=[])

    except Exception as e:
        return jsonify({"error": str(e)})
    
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
    return render_template('underconstrct.html')
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
