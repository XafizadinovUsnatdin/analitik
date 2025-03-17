import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
import matplotlib.pyplot as plt

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def train_model_for_product(product_id, df, time_steps=30):
    print(f"Training model for product ID: {product_id}")
    
    # Filter data for the specific product
    product_data = df[df['product_id'] == product_id].copy()
    
    # Convert date to datetime
    product_data['date'] = pd.to_datetime(product_data['date'], format='%d/%m/%Y')
    
    # Sort by date
    product_data = product_data.sort_values('date')
    
    # Filter data for training (2022-01-01 to 2024-12-31)
    train_data = product_data[
        (product_data['date'] >= '2022-01-01') & 
        (product_data['date'] <= '2024-12-31')
    ]
    
    if len(train_data) < 100:
        print(f"Not enough data for product {product_id}. Skipping.")
        return None
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_quantity = scaler.fit_transform(train_data['quantity'].values.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences(scaled_quantity, time_steps)
    
    # Reshape for LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Create and compile the model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(25),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    # Train the model
    history = model.fit(
        X, y, 
        epochs=20, 
        batch_size=32, 
        validation_split=0.2,
        verbose=1
    )
    
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = f'models/trained_model_product_{product_id}.keras'
    model.save(model_path)
    
    # Save the scaler
    np.save(f'models/scaler_product_{product_id}.npy', 
            {'data_min': scaler.data_min_, 'data_max': scaler.data_max_})
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss for Product {product_id}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Create directory if it doesn't exist
    os.makedirs('static/plots', exist_ok=True)
    plt.savefig(f'static/plots/training_history_product_{product_id}.png')
    plt.close()
    
    print(f"Model for product {product_id} trained and saved successfully.")
    return model_path

def main():
    # Load the sales data
    print("Loading sales data...")
    sales_df = pd.read_csv('data/sales_expanded_1000.csv')
    
    # Get unique product IDs
    product_ids = sales_df['product_id'].unique()
    
    # Train model for each product
    trained_models = {}
    for product_id in product_ids:
        model_path = train_model_for_product(product_id, sales_df)
        if model_path:
            trained_models[product_id] = model_path
    
    print(f"Training completed. {len(trained_models)} models trained.")

if __name__ == "__main__":
    main()

