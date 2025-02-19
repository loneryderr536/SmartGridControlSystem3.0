from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import asyncio
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = tf.keras.models.load_model("bi_lstm_model.keras")
scaler_load = joblib.load("scaler.pkl")

# Create thread pool for async operations
executor = ThreadPoolExecutor(max_workers=3)

@app.route('/')
def index():
    return render_template('index.html')
def load_data():
    """Load and preprocess the energy and weather data"""
    energy_data = pd.read_csv('data/energy_dataset.csv')
    weather_data = pd.read_csv('data/weather_features.csv')
    
    # Convert timestamps to datetime
    energy_data['time'] = pd.to_datetime(energy_data['time'], utc=True)
    energy_data.set_index('time', inplace=True)
    
    weather_data['dt_iso'] = pd.to_datetime(weather_data['dt_iso'], utc=True)
    weather_data.set_index('dt_iso', inplace=True)
    
    return energy_data, weather_data

# Load the data
energy_data, weather_data = load_data()

def prepare_sequence(scaled_data, sequence_length=24):
    """Prepare sequence for model prediction"""
    return scaled_data[-sequence_length:].reshape(1, sequence_length, -1)

async def predict_load(date, city_name):
    """Async function to predict load"""
    def prediction_task():
        try:
            # Convert input date to datetime
            input_date = pd.to_datetime(date, utc=True)
            
            # Prepare city encoding
            city_name = city_name.strip().lower()
            available_cities = pd.get_dummies(energy_data['city_name']).columns
            if city_name not in available_cities:
                raise ValueError(f"City not found. Available cities: {', '.join(available_cities)}")
            
            # Create city features
            city_features = np.zeros(len(available_cities))
            city_feature_index = list(available_cities).index(city_name)
            city_features[city_feature_index] = 1
            
            # Get the last sequence from the data
            last_sequence = energy_data['total load actual'].values[-24:]
            scaled_sequence = scaler_load.transform(last_sequence.reshape(-1, 1))
            
            # Make prediction
            X = prepare_sequence(scaled_sequence)
            prediction = model.predict(X)
            
            # Inverse transform the prediction
            predicted_load = scaler_load.inverse_transform(prediction)[0][0]
            
            return {
                'status': 'success',
                'predicted_load': float(predicted_load),
                'date': date,
                'city': city_name
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    # Run the prediction in the thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, prediction_task)
    return result

async def calculate_demand_response(date, city_name, strategy, percentage):
    """Async function to calculate demand response"""
    def dr_task():
        try:
            # First get the load prediction
            predicted_load = predict_load(date, city_name)
            
            # Apply demand response based on tool
            if strategy == "peak_reduction":
                # Reduce load during peak hours (assumed 18-22)
                reduction = float(percentage) / 100
                adjusted_load = predicted_load * (1 - reduction)
            elif strategy == "time_of_use":
                # Apply time-of-use pricing adjustments
                hour = pd.to_datetime(date).hour
                if 22 <= hour or hour < 6:  # Off-peak
                    adjusted_load = predicted_load * 0.8
                elif 18 <= hour < 22:  # Peak
                    adjusted_load = predicted_load * 1.1
                else:  # Normal
                    adjusted_load = predicted_load
            else:
                raise ValueError("Invalid strategy parameter")
            
            # Calculate price
            base_price = 10
            price_coefficient = 0.001
            price = base_price + price_coefficient * adjusted_load
            
            return {
                'status': 'success',
                'original_load': float(predicted_load),
                'adjusted_load': float(adjusted_load),
                'predicted_price': float(price),
                'date': date,
                'city': city_name,
                'strategy': strategy,
                'reduction_percentage': percentage
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    # Run the DR calculation in the thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, dr_task)
    return result

@app.route('http://127.0.0.1:5000/load-forecast-page', methods=['POST'])
async def generate_forecast():
    """API endpoint for load forecasting"""
    try:
        data = request.get_json()
        city_name = data.get('city_name')
        date = data.get('date')
        
        if not all([city_name, date]):
            return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400
        
        result = await predict_load(date, city_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('http://127.0.0.1:5000/demand-response', methods=['POST'])
async def generate_demand_response():
    """API endpoint for demand response"""
    try:
        data = request.get_json()
        city_name = data.get('city_name')
        date = data.get('date')
        strategy = data.get('strategy')
        percentage = data.get('percentage')
        
        if not all([city_name, date, strategy, percentage]):
            return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400
        
        result = await calculate_demand_response(date, city_name, strategy, percentage)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
