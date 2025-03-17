import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json
import os
from datetime import datetime, timedelta

def load_scaler(product_id):
    scaler_data = np.load(f'models/scaler_product_{product_id}.npy', allow_pickle=True).item()
    scaler = MinMaxScaler()
    scaler.data_min_ = scaler_data['data_min']
    scaler.data_max_ = scaler_data['data_max']
    return scaler

def forecast_for_product(product_id, sales_df, start_date, days_to_forecast, time_steps=30):
    print(f"Forecasting for product ID: {product_id}")
    
    try:
        # Load the model
        model = load_model(f'models/trained_model_product_{product_id}.keras')
        
        # Load the scaler
        scaler = load_scaler(product_id)
        
        # Filter data for the specific product
        product_data = sales_df[sales_df['product_id'] == product_id].copy()
        
        # Convert date to datetime
        product_data['date'] = pd.to_datetime(product_data['date'], format='%d/%m/%Y')
        
        # Sort by date
        product_data = product_data.sort_values('date')
        
        # Get the last time_steps days of data
        last_data = product_data.tail(time_steps)
        
        if len(last_data) < time_steps:
            print(f"Not enough historical data for product {product_id}. Skipping.")
            return None
        
        # Scale the data
        scaled_quantity = scaler.transform(last_data['quantity'].values.reshape(-1, 1))
        
        # Initialize the input sequence with the last time_steps values
        input_sequence = scaled_quantity.reshape(1, time_steps, 1)
        
        # Generate forecasts
        forecasts = []
        current_date = start_date
        
        for _ in range(days_to_forecast):
            # Predict the next value
            next_pred = model.predict(input_sequence, verbose=0)
            
            # Inverse transform to get the actual value
            next_pred_actual = scaler.inverse_transform(next_pred)[0][0]
            
            # Add to forecasts
            forecasts.append({
                "date": current_date.strftime('%Y-%m-%d'),
                "predicted_quantity": float(next_pred_actual)
            })
            
            # Update input sequence for next prediction
            input_sequence = np.append(input_sequence[:, 1:, :], 
                                      next_pred.reshape(1, 1, 1), 
                                      axis=1)
            
            # Move to next date
            current_date += timedelta(days=1)
        
        return {
            "product_id": int(product_id),
            "forecast": forecasts
        }
    
    except Exception as e:
        print(f"Error forecasting for product {product_id}: {e}")
        return None

def generate_monthly_forecasts(sales_df, year=2025, months=[1, 2, 3]):
    # Get unique product IDs
    product_ids = sales_df['product_id'].unique()
    
    # Create forecasts directory if it doesn't exist
    os.makedirs('data/forecasts', exist_ok=True)
    
    for month in months:
        # Define start date and days in month
        start_date = datetime(year, month, 1)
        
        # Calculate days in month
        if month == 12:
            next_month_start = datetime(year + 1, 1, 1)
        else:
            next_month_start = datetime(year, month + 1, 1)
        
        days_in_month = (next_month_start - start_date).days
        
        # Generate forecasts for each product
        monthly_forecasts = {}
        
        for product_id in product_ids:
            forecast = forecast_for_product(
                product_id, 
                sales_df, 
                start_date, 
                days_in_month
            )
            
            if forecast:
                monthly_forecasts[str(product_id)] = forecast
        
        # Save forecasts to JSON
        output_file = f'data/forecasts/forecast_{year}-{month:02d}.json'
        with open(output_file, 'w') as f:
            json.dump(monthly_forecasts, f, indent=4)
        
        print(f"Forecasts for {year}-{month:02d} saved to {output_file}")

def main():
    # Load the sales data
    print("Loading sales data...")
    sales_df = pd.read_csv('data/sales_expanded_1000.csv')
    
    # Generate forecasts for the first 3 months of 2025
    generate_monthly_forecasts(sales_df, year=2025, months=[1, 2, 3])
    
    print("Forecasting completed.")

if __name__ == "__main__":
    main()

