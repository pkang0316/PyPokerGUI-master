import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers

class QNetworkModel:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()
        # Make an initial prediction to initialize the model
        self.predict(np.zeros((1, state_size)))

    def build_model(self):
        input_layer = layers.Input(shape=(self.state_size,))
        hidden_layer = layers.Dense(64, activation='relu')(input_layer)
        hidden_layer = layers.Dense(64, activation='relu')(hidden_layer)
        output_layer = layers.Dense(self.action_size, activation='linear')(hidden_layer)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss='mse',
            optimizer='adam'
        )
        return model
    
    def predict(self, state):
        # Ensure state is correctly shaped
        state = np.array(state).reshape(1, -1)
        return self.model.predict(state, verbose=0)
    
    def update_weights(self, state, target_q_values):
        # Ensure state is correctly shaped
        state = np.array(state).reshape(1, -1)
        target_q_values = np.array(target_q_values)
        return self.model.fit(state, target_q_values, epochs=1, verbose=0)
    
    def save_model(self, filename):
        base_path = "players/neural_network/q_network"
        base_filename = filename.replace('.h5', '').replace('\\', '/')
        weights_filename = f"{base_path}/{base_filename}.weights.h5"
        print(f"Saving weights to: {weights_filename}")
        try:
            self.model.save_weights(weights_filename)
            print("Weights saved successfully")
        except Exception as e:
            print(f"Error saving weights: {e}")
    
    def load_model(self, filename):
        weights_filename = filename if filename.endswith('.weights.h5') else f"{filename}.weights.h5"
        print(f"Loading weights from: {weights_filename}")
        try:
            self.model.load_weights(weights_filename)
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Error loading weights: {e}")