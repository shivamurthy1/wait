import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns

# --------- CONFIG ------------
API_KEY = "8191e945a34549468e6123627251505"  # Your API key here
LOCATION = "New Delhi,India"  # Change as needed
DEFAULT_START_DATE = "2023-04-01"
DEFAULT_END_DATE = "2023-04-07"
# -----------------------------

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def fetch_weather_data(location, start_date, end_date, api_key):
    all_hours = []
    base_url = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx"
    
    for single_date in daterange(start_date, end_date):
        date_str = single_date.strftime("%Y-%m-%d")
        params = {
            'key': api_key,
            'q': location,
            'format': 'json',
            'date': date_str,
            'tp': '1',  # hourly data
        }
        print(f"Fetching data for {date_str}")
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} {response.text}")
        
        data = response.json()
        try:
            hours = data['data']['weather'][0]['hourly']
        except (KeyError, IndexError):
            print(f"No data for {date_str}, skipping...")
            continue
        
        for hour_data in hours:
            time_str = hour_data['time']
            hour = int(time_str) // 100 if len(time_str) > 2 else int(time_str) // 1
            
            record = {
                'date': date_str,
                'hour': hour,
                'tempC': float(hour_data['tempC']),
                'humidity': float(hour_data['humidity']),
                'pressure': float(hour_data['pressure']),
                'windspeedKmph': float(hour_data['windspeedKmph']),
            }
            all_hours.append(record)
        time.sleep(1)  # avoid rate limits
    
    return pd.DataFrame(all_hours)

def train_random_forest(df):
    X = df[['tempC', 'pressure', 'windspeedKmph', 'hour']]
    y = df['humidity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'R2 Score: {r2:.2f}')
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Humidity")
    plt.ylabel("Predicted Humidity")
    plt.title("Actual vs Predicted Humidity")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
    plt.show()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8,6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.title("Residuals Distribution")
    plt.show()
    
    # Feature Importance
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(8,6))
    feat_importances.sort_values().plot(kind='barh')
    plt.title("Feature Importances")
    plt.show()
    
    return model

def main():
    start_date_input = input(f"Enter start date (YYYY-MM-DD) or press Enter for default [{DEFAULT_START_DATE}]: ")
    end_date_input = input(f"Enter end date (YYYY-MM-DD) or press Enter for default [{DEFAULT_END_DATE}]: ")
    
    try:
        start_date = datetime.strptime(start_date_input, "%Y-%m-%d") if start_date_input else datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_input, "%Y-%m-%d") if end_date_input else datetime.strptime(DEFAULT_END_DATE, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format! Please enter dates in YYYY-MM-DD format.")
        return
    
    if end_date < start_date:
        print("End date cannot be earlier than start date.")
        return
    
    df = fetch_weather_data(LOCATION, start_date, end_date, API_KEY)
    print(f"Fetched {len(df)} records.")
    print(df.head())
    
    model = train_random_forest(df)
    
    sample = [[30.0, 1010, 10.0, 14]]  # tempC, pressure, windspeedKmph, hour
    predicted_humidity = model.predict(sample)
    print(f'Predicted Humidity for sample input: {predicted_humidity[0]:.2f}%')

if _name_ == "_main_":
    main()