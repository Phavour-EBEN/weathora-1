from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
from flask import send_from_directory
import requests
import random
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================================================
# CONFIGURATION
# ============================================================================

# Try both compatible and original model paths
NVAPS_MODEL_PATH = 'nvaps_lstm_model_compatible.h5'  # Use the fixed model
NVAPS_MODEL_PATH_FALLBACK = 'nvaps_lstm_model.h5'  # Fallback to original
NVAPS_SCALER_PATH = 'nvaps_scaler.pkl'
NVAPS_CONFIG_PATH = 'nvaps_config.json'
NVAPS_LABEL_ENCODER_PATH = 'nvaps_label_encoder.pkl'

# ============================================================================
# LOAD NVAPS MODEL AND CONFIGURATION
# ============================================================================

print("Loading NVAPS model and configurations...")
print(f"TensorFlow version: {tf.__version__}")

try:
    # Try loading the compatible model first
    model_path = NVAPS_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Compatible model not found, trying fallback...")
        model_path = NVAPS_MODEL_PATH_FALLBACK
    
    print(f"Loading model from: {model_path}")
    
    # Load model with compile=False to avoid optimizer/version issues
    nvaps_model = keras.models.load_model(
        model_path, 
        compile=False
    )
    
    # Recompile with basic settings (only needed for inference)
    nvaps_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Load auxiliary files
    with open(NVAPS_SCALER_PATH, 'rb') as f:
        nvaps_scaler = pickle.load(f)
    with open(NVAPS_CONFIG_PATH, 'r') as f:
        nvaps_config = json.load(f)
    with open(NVAPS_LABEL_ENCODER_PATH, 'rb') as f:
        nvaps_label_encoder = pickle.load(f)
    
    print("✓ NVAPS model loaded successfully")
    print(f"  Model input shape: {nvaps_model.input_shape}")
    print(f"  Model output shape: {nvaps_model.output_shape}")
    print(f"  Total parameters: {nvaps_model.count_params():,}")
    
except Exception as e:
    print(f"✗ NVAPS model failed to load: {e}")
    print(f"\nTroubleshooting:")
    print(f"  1. Ensure 'nvaps_lstm_model_compatible.h5' exists in current directory")
    print(f"  2. Check TensorFlow version: pip install tensorflow==2.19.0")
    print(f"  3. Run the compatibility fixer script in Colab")
    nvaps_model = None
    nvaps_scaler = None
    nvaps_config = None
    nvaps_label_encoder = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_input_sequence_nvaps(historical_data, lookback, feature_cols, scaler_obj):
    """
    Prepare input sequence for NVAPS model with rolling features.
    
    Parameters:
    -----------
    historical_data : list of dicts
        Historical TPW data points
    lookback : int
        Number of past observations to use
    feature_cols : list
        List of feature column names
    scaler_obj : sklearn scaler
        Fitted scaler object
    
    Returns:
    --------
    numpy array : Scaled input sequence ready for prediction
    """
    df = pd.DataFrame(historical_data)
    
    # Add temporal features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Add rolling features (CRITICAL - NVAPS model expects these!)
    for window in [7, 14, 30]:
        df[f'tpw_roll_mean_{window}d'] = df['tpw_mean'].rolling(window, min_periods=1).mean()
        df[f'tpw_roll_std_{window}d'] = df['tpw_mean'].rolling(window, min_periods=1).std()
        df[f'tpw_trend_{window}d'] = df['tpw_mean'] - df[f'tpw_roll_mean_{window}d']
    
    # Fill NaN values from rolling stats
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Select features and take last lookback days
    features = df[feature_cols].values[-lookback:]
    
    if len(features) < lookback:
        raise ValueError(f"Need at least {lookback} days of data, got {len(features)}")
    
    # Scale features
    features_scaled = scaler_obj.transform(features)
    
    # Reshape for model input: (1, lookback, n_features)
    return features_scaled.reshape(1, lookback, len(feature_cols))

def tpw_to_weather_estimates(tpw_value, month=None, region_temp_avg=27):
    """
    Derive weather parameter estimates from TPW value.
    This is an approximation based on atmospheric physics relationships.
    
    Parameters:
    -----------
    tpw_value : float
        Total Precipitable Water in mm
    month : int
        Month of year (1-12) for seasonal adjustments
    region_temp_avg : float
        Average temperature for the region (default 27°C for Ghana)
    
    Returns:
    --------
    dict : Estimated weather parameters
    """
    # TPW correlations (empirically derived for tropical regions)
    
    # Humidity: Strong positive correlation with TPW
    # TPW range ~20-70mm maps to ~40-95% humidity
    humidity = min(95, max(40, 40 + (tpw_value - 20) * 1.1))
    
    # Cloud cover: Higher TPW = more clouds
    if tpw_value < 30:
        cloudcover = 10 + (tpw_value - 20) * 2
    elif tpw_value < 50:
        cloudcover = 30 + (tpw_value - 30) * 2
    else:
        cloudcover = 70 + min(30, (tpw_value - 50) * 0.5)
    cloudcover = max(0, min(100, cloudcover))
    
    # Precipitation: Higher TPW increases rain probability
    if tpw_value < 35:
        precip = 0
    elif tpw_value < 50:
        precip = random.uniform(0, 2)
    elif tpw_value < 60:
        precip = random.uniform(1, 8)
    else:
        precip = random.uniform(5, 20)
    
    # Pressure: Inverse relationship with humidity/TPW
    pressure = 1013 - (humidity - 50) * 0.15
    
    # Temperature adjustment based on moisture
    temp = region_temp_avg + (humidity - 70) * 0.1
    feelslike = temp + (humidity - 60) * 0.15
    
    # Wind: Generally higher with unstable moist air
    if tpw_value < 40:
        wind_speed = random.uniform(5, 15)
    elif tpw_value < 55:
        wind_speed = random.uniform(10, 25)
    else:
        wind_speed = random.uniform(15, 35)
    
    wind_degree = random.randint(0, 359)
    wind_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    wind_dir = wind_dirs[int((wind_degree + 22.5) / 45) % 8]
    
    # UV index: Affected by cloud cover
    uv_base = 7 if month and month in [3,4,5,6,7,8] else 5  # Higher in dry season
    uv_index = max(0, uv_base * (1 - cloudcover/150))
    
    return {
        'tpw_mm': round(tpw_value, 2),
        'humidity': round(humidity, 0),
        'cloudcover': round(cloudcover, 0),
        'precip_mm': round(precip, 1),
        'pressure_mb': round(pressure, 0),
        'temperature_c': round(temp, 1),
        'feelslike_c': round(feelslike, 1),
        'wind_speed_kmh': round(wind_speed, 1),
        'wind_degree': wind_degree,
        'wind_dir': wind_dir,
        'uv_index': round(uv_index, 1),
        'estimation_note': 'Derived from NVAPS TPW model predictions'
    }

def fetch_current_weather(lat=5.6037, lon=-0.1870, api_key=None):
    """
    Fetch current weather from external API.
    Default coordinates are for Accra, Ghana.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    api_key : str
        API key for weather service
    
    Returns:
    --------
    dict : Current weather data
    """
    if not api_key:
        # Return mock data if no API key
        return {
            'source': 'mock',
            'location': f'Ghana ({lat:.2f}, {lon:.2f})',
            'temperature_c': 28.5,
            'feelslike_c': 32.1,
            'humidity': 78,
            'cloudcover': 40,
            'pressure_mb': 1011,
            'precip_mm': 0,
            'wind_speed_kmh': 15.2,
            'wind_degree': 221,
            'wind_dir': 'SW',
            'uv_index': 6,
            'condition': 'Partly cloudy',
            'note': 'Mock data - provide API key for real data'
        }
    
    try:
        # WeatherAPI.com
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            current = data['current']
            
            return {
                'source': 'weatherapi.com',
                'location': data['location']['name'],
                'temperature_c': current['temp_c'],
                'feelslike_c': current['feelslike_c'],
                'humidity': current['humidity'],
                'cloudcover': current['cloud'],
                'pressure_mb': current['pressure_mb'],
                'precip_mm': current['precip_mm'],
                'wind_speed_kmh': current['wind_kph'],
                'wind_degree': current['wind_degree'],
                'wind_dir': current['wind_dir'],
                'uv_index': current['uv'],
                'condition': current['condition']['text']
            }
    except Exception as e:
        return {'error': str(e), 'source': 'api_error'}

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """API home endpoint with documentation."""
    return jsonify({
        'message': 'NVAPS Weather Prediction API',
        'version': '1.0',
        'model': 'NVAPS LSTM',
        'status': 'online' if nvaps_model is not None else 'model not loaded',
        'endpoints': {
            '/': 'API documentation (this page)',
            '/health': 'Health check',
            '/model/info': 'Model information (GET)',
            '/predict': 'TPW prediction (POST)',
            '/weather/predict': 'Weather forecast from TPW (POST)',
            '/weather/current': 'Current weather + forecast (GET/POST)',
            '/dashboard': 'Visual dashboard'
        },
        'weather_parameters': [
            'temperature_c', 'feelslike_c', 'humidity', 'cloudcover',
            'pressure_mb', 'precip_mm', 'wind_speed_kmh', 'wind_degree',
            'wind_dir', 'uv_index'
        ],
        'documentation': 'Send historical TPW data to /predict or /weather/predict for forecasts'
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if nvaps_model is not None else 'unhealthy',
        'model_loaded': nvaps_model is not None,
        'scaler_loaded': nvaps_scaler is not None,
        'config_loaded': nvaps_config is not None,
        'label_encoder_loaded': nvaps_label_encoder is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get NVAPS model information and configuration."""
    if nvaps_config is None:
        return jsonify({'error': 'NVAPS model not loaded'}), 500
    
    return jsonify({
        'model_name': 'NVAPS LSTM',
        'model_type': 'LSTM Neural Network',
        'target_variable': nvaps_config.get('lstm_config', {}).get('target_var'),
        'lookback_period': nvaps_config.get('lstm_config', {}).get('lookback'),
        'feature_columns': nvaps_config.get('lstm_config', {}).get('feature_cols'),
        'model_path': NVAPS_MODEL_PATH,
        'description': 'TPW prediction model using NVAPS satellite data'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make TPW prediction using NVAPS model.
    
    Expected JSON format:
    {
        "historical_data": [
            {
                "tpw_mean": 45.2,
                "date": "2024-01-01"
            },
            ... (need at least 'lookback' days of data)
        ],
        "forecast_days": 7  // optional, defaults to 7
    }
    
    Returns:
    {
        "predicted_tpw_mm": 46.5,
        "forecast_date": "2024-01-08",
        "metadata": {...}
    }
    """
    if nvaps_model is None:
        return jsonify({'error': 'NVAPS model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'historical_data' not in data:
            return jsonify({'error': 'Missing historical_data in request'}), 400
        
        historical_data = data['historical_data']
        forecast_days = data.get('forecast_days', 7)
        
        # Get model configuration
        lookback = nvaps_config['lstm_config']['lookback']
        feature_cols = nvaps_config['lstm_config']['feature_cols']
        target_var = nvaps_config['lstm_config']['target_var']
        
        # Validate data length
        if len(historical_data) < lookback:
            return jsonify({
                'error': f'Need at least {lookback} days of historical data, got {len(historical_data)}'
            }), 400
        
        # Prepare input with rolling features
        input_sequence = prepare_input_sequence_nvaps(
            historical_data, lookback, feature_cols, nvaps_scaler
        )
        
        # Make prediction
        pred_scaled = nvaps_model.predict(input_sequence, verbose=0)[0, 0]
        
        # Inverse transform to get actual TPW value
        target_idx = feature_cols.index(target_var)
        dummy_array = np.zeros((1, len(feature_cols)))
        dummy_array[0, target_idx] = pred_scaled
        tpw_prediction = nvaps_scaler.inverse_transform(dummy_array)[0, target_idx]
        
        # Calculate forecast date
        last_date = pd.to_datetime(historical_data[-1].get('date', datetime.now()))
        forecast_date = last_date + timedelta(days=forecast_days)
        
        # Prepare response
        response = {
            'predicted_tpw_mm': round(float(tpw_prediction), 2),
            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
            'target_variable': target_var,
            'metadata': {
                'last_data_date': last_date.strftime('%Y-%m-%d'),
                'lookback_days': lookback,
                'data_points_used': len(historical_data),
                'forecast_days_ahead': forecast_days,
                'model': 'NVAPS LSTM'
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/predict', methods=['POST'])
def weather_predict():
    """
    Get weather forecast based on TPW prediction.
    
    Expected JSON format:
    {
        "historical_data": [
            {"tpw_mean": 45.2, "date": "2024-01-01"},
            ...
        ],
        "forecast_days": 7,  // optional
        "location": {  // optional
            "name": "Accra",
            "lat": 5.6037,
            "lon": -0.1870
        }
    }
    
    Returns weather forecast derived from TPW prediction.
    """
    if nvaps_model is None:
        return jsonify({'error': 'NVAPS model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'historical_data' not in data:
            return jsonify({'error': 'Missing historical_data'}), 400
        
        historical_data = data['historical_data']
        forecast_days = data.get('forecast_days', 7)
        location = data.get('location', {'name': 'Ghana', 'lat': 5.6037, 'lon': -0.1870})
        
        # Get model configuration
        lookback = nvaps_config['lstm_config']['lookback']
        feature_cols = nvaps_config['lstm_config']['feature_cols']
        target_var = nvaps_config['lstm_config']['target_var']
        
        # Prepare input with rolling features
        input_sequence = prepare_input_sequence_nvaps(
            historical_data, lookback, feature_cols, nvaps_scaler
        )
        
        # Predict TPW
        pred_scaled = nvaps_model.predict(input_sequence, verbose=0)[0, 0]
        
        # Inverse transform
        target_idx = feature_cols.index(target_var)
        dummy_array = np.zeros((1, len(feature_cols)))
        dummy_array[0, target_idx] = pred_scaled
        tpw_prediction = nvaps_scaler.inverse_transform(dummy_array)[0, target_idx]
        
        # Calculate forecast date
        last_date = pd.to_datetime(historical_data[-1].get('date', datetime.now()))
        forecast_date = last_date + timedelta(days=forecast_days)
        
        # Convert TPW to weather parameters
        weather_estimates = tpw_to_weather_estimates(tpw_prediction, month=forecast_date.month)
        
        return jsonify({
            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
            'location': location,
            'weather': weather_estimates,
            'tpw_data': {
                'predicted_tpw_mm': round(float(tpw_prediction), 2),
                'last_tpw_mm': round(float(historical_data[-1]['tpw_mean']), 2),
                'change': round(float(tpw_prediction - historical_data[-1]['tpw_mean']), 2)
            },
            'metadata': {
                'model': 'NVAPS LSTM',
                'forecast_days_ahead': forecast_days,
                'last_data_date': last_date.strftime('%Y-%m-%d')
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/current', methods=['GET', 'POST'])
def weather_current():
    """
    Get current weather conditions, optionally enhanced with TPW prediction.
    
    Query params or JSON:
    {
        "lat": 5.6037,
        "lon": -0.1870,
        "api_key": "your_key",  // optional
        "include_forecast": true  // optional
    }
    """
    try:
        # Get parameters
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = {}
        
        lat = float(data.get('lat', request.args.get('lat', 5.6037)))
        lon = float(data.get('lon', request.args.get('lon', -0.1870)))
        api_key = data.get('api_key', request.args.get('api_key'))
        include_forecast = data.get('include_forecast', 
                                   request.args.get('include_forecast', 'false').lower() == 'true')
        
        # Fetch current weather
        current_weather = fetch_current_weather(lat, lon, api_key)
        
        response = {
            'current': current_weather,
            'location': {
                'lat': lat,
                'lon': lon
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add TPW-based forecast if requested
        if include_forecast and nvaps_model is not None:
            try:
                # Use current humidity to estimate TPW
                if 'humidity' in current_weather:
                    humidity = current_weather['humidity']
                    estimated_tpw = 20 + (humidity - 40) / 1.1
                    
                    # Simple projection (would need actual historical data for real prediction)
                    future_tpw = estimated_tpw * 1.05
                    
                    forecast_weather = tpw_to_weather_estimates(
                        future_tpw,
                        month=(datetime.now() + timedelta(days=7)).month
                    )
                    
                    response['forecast_7day'] = {
                        'date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                        'weather': forecast_weather,
                        'note': 'Simplified TPW-based forecast - use /weather/predict with historical data for accurate predictions'
                    }
            except Exception as e:
                response['forecast_error'] = str(e)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Serve dashboard HTML."""
    return send_from_directory('.', 'dashboard.html')

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌤️  NVAPS WEATHER PREDICTION API SERVER")
    print("="*60)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /              - API documentation")
    print("  GET  /health        - Health check")
    print("  GET  /model/info    - NVAPS model information")
    print("  POST /predict       - TPW prediction")
    print("  POST /weather/predict - Weather forecast from TPW")
    print("  GET  /weather/current - Current weather + forecast")
    print("  GET  /dashboard     - Visual dashboard")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)