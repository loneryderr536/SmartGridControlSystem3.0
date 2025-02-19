import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
import joblib  # To load scaler

# Initialize Flask app
app = Flask(__name__)

# Load trained Bi-LSTM model
model = tf.keras.models.load_model("bi_lstm_model.keras")

X_train = np.random.rand(100, 13)

# Assuming X_train is your training data with 13 features
scaler = MinMaxScaler()
scaler.fit(X_train)  # X_train should have shape (n_samples, 13)

# Save the scaler
joblib.dump(scaler, "feature_scaler.pkl")
# Example: Create dummy target data (replace this with your actual target data)
y_train = np.random.rand(100, 1)  # 100 samples, 1 feature (the load values)

# Initialize and fit the target scaler
target_scaler = MinMaxScaler()
target_scaler.fit(y_train)

# Save the fitted scaler to a file
joblib.dump(target_scaler, "target_scaler.pkl")
# Load MinMaxScaler (ensure it's the same one used during training)
scaler = joblib.load("scaler.pkl")
# Load the correct scalers
feature_scaler = joblib.load("feature_scaler.pkl")  # fitted on the 13 input features
target_scaler = joblib.load("target_scaler.pkl")    # fitted on the target load (1 feature)

def predict_load(input_date, city, predict_past=False):
    input_date = pd.to_datetime(input_date, utc=True)

    # Validate the input date (keep your existing logic)
    if predict_past:
        if input_date >= energy_data.index[0]:
            raise ValueError("Input date must be before the dataset start date.")
    else:
        if input_date <= energy_data.index[-1]:
            raise ValueError("Input date must be in the future.")

    # Use the city names as used in training (all lowercase)
    city = city.strip().lower()
    # Define a mapping based on the training data; for 5 cities, for example:
    city_mapping = {'valencia': 0, 'bilbao': 1, 'seville': 2, 'madrid': 3, 'barcelona': 4}
    if city not in city_mapping:
        raise ValueError(f"City name not found. Available cities: {', '.join(city_mapping.keys())}")

    # Create a one-hot vector for the city with length matching training (5 in this case)
    city_encoded = np.zeros((1, len(city_mapping)), dtype=np.float32)
    city_encoded[0, city_mapping[city]] = 1.0

    # Instead of creating a new input from scratch, use the last sequence from your training data
    # which has the correct shape (sequence_length, 6)
    future_sequence = scaled_data[-sequence_length:, :].copy()

    # Determine how many days you need to predict (if applicable)
    days_to_predict = abs((input_date - energy_data.index[-1]).days)

    # For each day, do an iterative prediction
    for _ in range(days_to_predict):
        # The model expects input of shape (batch_size, sequence_length, features)
        predicted_scaled = model.predict(future_sequence.reshape(1, sequence_length, -1))
        # Create a new step: predicted load is the first element and append the city features
        new_step = np.hstack((predicted_scaled, city_encoded))
        # Update the sequence: remove the oldest timestep and add the new step
        future_sequence = np.vstack((future_sequence[1:], new_step))

    # Inverse scale the predicted load
    predicted_load = scaler_load.inverse_transform(predicted_scaled.reshape(-1, 1))
    return float(predicted_load[0, 0])

# Demand Response Adjustment Function
def apply_demand_response(predicted_load, strategy="peak_reduction", reduction_percent=20):
    """
    Adjusts the predicted load based on demand response strategies.
    """
    adjusted_load = predicted_load

    if strategy == "peak_reduction":
        adjusted_load *= (1 - reduction_percent / 100)  # Reduce peak load by percentage

    elif strategy == "time_of_use":
        # Simulate pricing effect: Off-peak reduction, Peak increase
        adjusted_load *= (0.9 if reduction_percent > 10 else 1.1)  

    return adjusted_load

# (Other classes and functions omitted for brevity...)

# Flask Route: Homepage (Dashboard)
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Flask Route: Load Forecast Page
@app.route("/load-forecast-page", methods=["GET", "POST"])
def load_forecast():
    if request.method == "POST":
        try:
            date = request.form["date"]
            city = request.form["city"]

            # Get forecasted load
            predicted_load = predict_load(date, city)

            return render_template("load-forecast-page.html", 
                                   city=city, 
                                   prediction=(date, predicted_load))
        except Exception as e:
            return render_template("load-forecast-page.html", 
                                   error=str(e))

    return render_template("load-forecast-page.html")

# Flask Route: Demand Response Page
@app.route("/demand-response-page", methods=["GET", "POST"])
def demand_response_page():
    if request.method == "POST":
        date = request.form["date"]
        city = request.form["city"]
        strategy = request.form["strategy"]
        reduction_percent = int(request.form["reduction"])

        # Get forecasted load
        predicted_load = predict_load(date, city)

        # Apply demand response
        adjusted_load = apply_demand_response(predicted_load, strategy, reduction_percent)

        return render_template("demand-response-page.html", 
                               city=city, 
                               prediction=(date, predicted_load, adjusted_load))

    return render_template("demand-response-page.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
