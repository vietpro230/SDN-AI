import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model as tf_load_model
    from tensorflow.keras.layers import Dense, LSTM
    HAS_TF = True
except ImportError:
    print("TensorFlow not found. Using sklearn MLPRegressor as fallback.")
    from sklearn.neural_network import MLPRegressor
    HAS_TF = False

class TrafficPredictor:
    def __init__(self, look_back=5, n_features=1):
        """
        AI Model for Traffic Prediction.

        This model predicts future network traffic loads to enable the SDN Controller
        to perform Energy Efficiency Optimization (Dynamic Resource Allocation).

        look_back: Number of past time steps to use for prediction.
        n_features: Number of switches (features) to predict.
        """
        self.look_back = look_back
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - self.look_back):
            a = dataset[i:(i + self.look_back), :]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back, :])
        return np.array(dataX), np.array(dataY)

    def train(self, df):
        # df is a DataFrame where columns are switches and rows are time steps
        self.n_features = df.shape[1]
        data = df.values
        data = self.scaler.fit_transform(data)

        X, y = self.create_dataset(data)

        if HAS_TF:
            # X shape: [samples, time steps, features]
            # y shape: [samples, features]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

            self.model = Sequential()
            self.model.add(LSTM(100, input_shape=(self.look_back, self.n_features)))
            self.model.add(Dense(self.n_features)) # Output layer has 'n_features' neurons (one per switch)
            self.model.compile(loss='mean_squared_error', optimizer='adam')

            print(f"Training TensorFlow model for {self.n_features} switches...")
            self.model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)

            train_score = self.model.evaluate(X_train, y_train, verbose=0)
            print(f'Train Score: {train_score:.4f} MSE')
        else:
            # Flatten X for MLP: [samples, look_back * features]
            n_samples = X.shape[0]
            X_flat = X.reshape(n_samples, -1)

            X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42, shuffle=False)

            self.model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
            print(f"Training MLPRegressor model for {self.n_features} switches...")
            self.model.fit(X_train, y_train)
            print(f'Train Score: {self.model.score(X_train, y_train):.2f} R2')

    def predict(self, recent_data):
        """
        Predicts the next traffic value for ALL switches.
        recent_data: array of shape (look_back, n_features)
        """
        if self.model is None:
            raise Exception("Model not trained yet.")

        # Scale input
        # Note: recent_data should be raw values, we need to scale them
        # But wait, scaler expects (n_samples, n_features).
        # We have (look_back, n_features).
        input_scaled = self.scaler.transform(recent_data)

        if HAS_TF:
            input_data = input_scaled.reshape(1, self.look_back, self.n_features)
            prediction_scaled = self.model.predict(input_data)
        else:
            input_data = input_scaled.reshape(1, -1)
            prediction_scaled = self.model.predict(input_data)

        # Inverse scale output
        prediction = self.scaler.inverse_transform(prediction_scaled)
        return prediction[0]

    def save_model(self, path_prefix):
        """
        Saves the trained model and scaler to disk.
        path_prefix: The base path (without extension) to save files.
        """
        if self.model is None:
            raise Exception("Model not trained yet.")

        # Save Scaler
        joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")

        # Save Model
        if HAS_TF and isinstance(self.model, Sequential):
            self.model.save(f"{path_prefix}_model.h5")
            print(f"Model saved to {path_prefix}_model.h5")
        else:
            joblib.dump(self.model, f"{path_prefix}_model.pkl")
            print(f"Model saved to {path_prefix}_model.pkl")

    def load_model(self, path_prefix):
        """
        Loads the trained model and scaler from disk.
        path_prefix: The base path (without extension) to load files from.
        """
        # Load Scaler
        scaler_path = f"{path_prefix}_scaler.pkl"
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        # Load Model
        if HAS_TF and os.path.exists(f"{path_prefix}_model.h5"):
            self.model = tf_load_model(f"{path_prefix}_model.h5")
            print(f"TensorFlow model loaded from {path_prefix}_model.h5")
        elif os.path.exists(f"{path_prefix}_model.pkl"):
            self.model = joblib.load(f"{path_prefix}_model.pkl")
            print(f"Scikit-learn model loaded from {path_prefix}_model.pkl")
        else:
            raise FileNotFoundError(f"Model file not found at {path_prefix}_model.h5 or .pkl")
