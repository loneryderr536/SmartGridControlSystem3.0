from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib
matplotlib.use("Agg")  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import base64
import io
import os
from datetime import datetime

# Create Flask application instance
app = Flask(__name__)

# Global variables to store model and data preprocessing objects
model = None
scaler_load = None
scaler_city = None
energy_data = None
city_encoded = None
scaled_data = None
sequence_length = 24



#open via index page
@app.route('/')
def index():
    return render_template('index.html')

def initialize_model():
    global model, scaler_load, scaler_city, energy_data, city_encoded, scaled_data
    
    try:
        # Load the datasets
        file_path_energy = os.path.join('data', 'energy_dataset.csv')
        file_path_weather = os.path.join('data', 'weather_features.csv')
        
        # Verify files exist
        if not os.path.exists(file_path_energy):
            raise FileNotFoundError(f"Energy dataset not found at {file_path_energy}")
        if not os.path.exists(file_path_weather):
            raise FileNotFoundError(f"Weather dataset not found at {file_path_weather}")

        # Load datasets
        energy_data = pd.read_csv(file_path_energy)
        weather_data = pd.read_csv(file_path_weather)

        # Data preprocessing for energy data
        energy_data['time'] = pd.to_datetime(energy_data['time'])
        energy_data.set_index('time', inplace=True)

        # Data preprocessing for weather data
        weather_data['dt_iso'] = pd.to_datetime(weather_data['dt_iso'])
        weather_data.set_index('dt_iso', inplace=True)

        # Merge datasets
        combined_data = pd.merge(energy_data, weather_data, 
                               left_index=True, right_index=True, 
                               how='inner')

        # Handle missing values
        combined_data.dropna(subset=['total load actual'], inplace=True)
        
        # Ensure city_name column exists and process it
        if 'city_name' not in combined_data.columns:
            raise ValueError("city_name column not found in the dataset")
        
        combined_data['city_name'] = combined_data['city_name'].fillna('unknown')
        combined_data['city_name'] = combined_data['city_name'].str.strip().str.lower()

        # Create city encoding
        city_encoded = pd.get_dummies(combined_data['city_name'])
        if city_encoded.empty:
            raise ValueError("No cities found in the dataset")

        print(f"Available cities: {city_encoded.columns.tolist()}")  # Debug print

        # Scale the load data
        scaler_load = MinMaxScaler(feature_range=(0, 1))
        scaled_load = scaler_load.fit_transform(combined_data[['total load actual']])

        # Scale the city encoding
        scaler_city = MinMaxScaler(feature_range=(0, 1))
        scaled_city = scaler_city.fit_transform(city_encoded)

        # Combine scaled data
        scaled_data = np.hstack((scaled_load, scaled_city))

        # Prepare sequences for LSTM
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Build the model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, 
                               input_shape=(sequence_length, X.shape[2])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        print("Model initialization completed successfully")
        return model, scaler_load, scaler_city, energy_data, city_encoded, scaled_data

    except Exception as e:
        print(f"Error in initialize_model: {str(e)}")
        raise

@app.route('/load-forecast-page', methods=['GET', 'POST'])
def load_forecast():
    global model, city_encoded, energy_data, scaler_load, scaler_city, scaled_data
    
    # Get current date for form validation
    current_date = datetime.now().date().strftime('%Y-%m-%d')
    
    try:
        # Initialize model if not already initialized
        if model is None or city_encoded is None:
            print("Initializing model...")
            model, scaler_load, scaler_city, energy_data, city_encoded, scaled_data = initialize_model()
            print("Model initialization complete")

        available_cities = sorted([city.lower() for city in city_encoded.columns])
        print(f"Available cities: {available_cities}")
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        return render_template('load-forecast-page.html',
                            error=f"Error initializing model: {str(e)}",
                            available_cities=['valencia', 'barcelona', 'madrid', 'seville', 'bilbao'],
                            current_date=current_date)

    if request.method == 'POST':
        try:
            user_input_date = request.form.get('date')
            user_input_city = request.form.get('text', '').strip().lower()
            
            # Input validation
            if not user_input_date or not user_input_city:
                raise ValueError("Please provide both date and city.")
            
            # Generate predictions
            predictions, dates = predict_future_load(user_input_date, user_input_city)
            
            # Get only the prediction for the requested date
            target_date = pd.to_datetime(user_input_date).date()
            prediction = None
            
            for i, date in enumerate(dates):
                # Convert datetime to date for comparison if it's not already a date
                date_to_compare = date.date() if hasattr(date, 'date') else date
                if date_to_compare == target_date:
                    prediction = round(float(predictions[i]), 2)
                    break
            
            if prediction is None:
                raise ValueError("No prediction available for the specified date.")
            
            # Format the single prediction for display
            formatted_prediction = (user_input_date, prediction)
            
            return render_template('load-forecast-page.html',
                                prediction=formatted_prediction,  # Single prediction
                                city=user_input_city.title(),
                                available_cities=available_cities,
                                current_date=current_date)
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return render_template('load-forecast-page.html',
                                error=str(e),
                                available_cities=available_cities,
                                current_date=current_date)
    
    # For GET request
    return render_template('load-forecast-page.html',
                         available_cities=available_cities,
                         current_date=current_date)     
def predict_future_load(input_date, city_name):
    """
    Predict future load for a given date and city.
    """
    print(f"Starting prediction for date: {input_date}, city: {city_name}")  # Debug print
    
    try:
        # Convert input to date object
        input_date = pd.to_datetime(input_date).date()
        print(f"Parsed input date: {input_date}")  # Debug print
        
        # Get current date (without time)
        current_date = datetime.now().date()
        print(f"Current date: {current_date}")  # Debug print
        
        # Get the last date from energy data
        last_training_date = pd.to_datetime(energy_data.index[-1]).date()
        print(f"Last training date: {last_training_date}")  # Debug print

        # Validate dates
        if input_date <= current_date:
            raise ValueError("Please select a future date.")

        if input_date <= last_training_date:
            raise ValueError(f"Input date must be after {last_training_date.strftime('%Y-%m-%d')}")

        # Process city name
        city_name = city_name.strip().lower()
        
        # Verify city exists in encoded data
        if city_name not in city_encoded.columns:
            raise ValueError(f"Invalid city. Available cities: {', '.join(sorted(city_encoded.columns))}")
        
        # Get city feature index
        city_feature_index = list(city_encoded.columns).index(city_name)
        
        # Prepare city features
        city_feature_values = np.zeros(len(city_encoded.columns))
        city_feature_values[city_feature_index] = 1
        city_feature_values = city_feature_values.reshape(1, -1)
        
        # Scale city features
        city_feature_values_scaled = scaler_city.transform(
            pd.DataFrame(city_feature_values, columns=city_encoded.columns))

        # Calculate days to predict
        days_to_predict = (input_date - last_training_date).days
        print(f"Days to predict: {days_to_predict}")  # Debug print
        
        if days_to_predict <= 0:
            raise ValueError("Input date must be in the future.")

        # Initialize prediction variables
        predictions = []
        dates = []
        current_pred_date = last_training_date

        # Generate predictions
        future_sequence = scaled_data[-sequence_length:, :].copy()
        
        for i in range(days_to_predict):
            # Make prediction
            predicted_scaled = model.predict(
                future_sequence.reshape(1, sequence_length, -1), 
                verbose=0
            )
            
            # Prepare next sequence
            city_feature_values_scaled_tiled = np.tile(
                city_feature_values_scaled, (predicted_scaled.shape[0], 1))
            new_step = np.hstack((predicted_scaled, city_feature_values_scaled_tiled))
            future_sequence = np.vstack((future_sequence[1:], new_step))
            
            # Update date and store results
            current_pred_date += pd.Timedelta(days=1)
            dates.append(current_pred_date)
            
            # Convert prediction back to original scale
            prediction = float(scaler_load.inverse_transform(
                predicted_scaled.reshape(-1, 1))[0, 0])
            predictions.append(prediction)
            
            print(f"Generated prediction {i+1}/{days_to_predict}")  # Debug print


        print(f"Successfully generated {len(predictions)} predictions")  # Debug print
        return predictions, dates

    except Exception as e:
        print(f"Error in predict_future_load: {str(e)}")  # Debug print
        raise


@app.route('/fault-detection-page', methods=['GET', 'POST'])
def fault_detection():
    return render_template('fault-detection-page.html')


if __name__ == '__main__':
    app.run(debug=True)