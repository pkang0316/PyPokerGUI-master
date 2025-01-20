import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json

class QNetworkModel:
    def __init__(self, action_size):
        """
        Initialize the Q-Network model for poker decision making.
        
        The model processes three types of input:
        1. Hole cards (private cards)
        2. Community cards (shared cards)
        3. Numeric features (pot, stack sizes, position, equity)
        
        Args:
            action_size (int): Number of possible actions (typically fold, call, raise)
        """
        # Model identification
        self.model_name = 'q_network_model'
        self.version = "1.0"
        
        # Initialize performance tracking
        self.training_history = {
            'timestamp': [],       # When the data point was recorded
            'episode': [],         # Training episode number
            'loss': [],           # Training loss
            'action_taken': [],    # What action was chosen
            'reward': [],         # Reward received
            'state_info': [],     # Key state information (e.g., hand type, position)
            'q_values': []        # Model's Q-value predictions
        }
        
        # Load configuration and set up directories
        self.config = self._load_config()
        self.directories = self._setup_directories()
        
        # Define the dimensions of our state space components
        self.hole_cards_dim = 52      # One-hot encoding for 2 cards
        self.community_cards_dim = 52  # One-hot encoding for up to 5 community cards
        self.numeric_features_dim = 5  # pot, stack, position, opponent_stack, equity
        
        # Calculate total state size and store action size
        self.state_size = self.hole_cards_dim + self.community_cards_dim + self.numeric_features_dim
        self.action_size = action_size
        
        # Build the neural network model
        self.model = self.build_model()

    def _load_config(self):
        """Load model configuration from JSON file."""
        config_path = Path(__file__).parent.parent / 'model_conf.json'
        
        try:
            # Use utf-8-sig to handle potential BOM in the JSON file
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                config = json.load(f)
                
            if not config or 'directories' not in config:
                raise ValueError("Configuration missing required 'directories' section")
                
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model configuration: {str(e)}")

    def _setup_directories(self):
        """
        Create and verify the directory structure for model storage.
        
        Creates directories for:
        - Base models
        - Saved models
        - Q-network specific files
        - Model weights
        - Training checkpoints
        """
        dirs = {}
        
        try:
            # Set up base model directory
            dirs['base'] = Path(self.config['directories']['base_models'])
            
            # Set up model-specific directories
            model_config = self.config['directories']['model']
            dirs['saved_models'] = Path(model_config['saved_models'])
            
            # Set up Q-network specific directories
            qnet_config = model_config['q_network']
            dirs['q_network'] = Path(qnet_config['base'])
            dirs['weights'] = Path(qnet_config['weights'])
            dirs['checkpoints'] = Path(qnet_config['checkpoints'])
            
            # Create all directories if they don't exist
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)
                
            return dirs
            
        except KeyError as e:
            raise RuntimeError(f"Missing required directory configuration: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to set up directories: {str(e)}")

    def build_model(self):
        """
        Construct the neural network architecture for the Q-network.
        
        The network processes three input streams:
        1. Hole cards through a dense network
        2. Community cards through a dense network
        3. Numeric features through a separate dense network
        
        These are then concatenated and processed through final dense layers
        to produce Q-values for each possible action.
        """
        # Input layers for different components of the poker state
        hole_cards_input = keras.layers.Input(shape=(self.hole_cards_dim,), name='hole_cards')
        community_cards_input = keras.layers.Input(shape=(self.community_cards_dim,), name='community_cards')
        numeric_input = keras.layers.Input(shape=(self.numeric_features_dim,), name='numeric_features')
        
        # Process hole cards
        hole_cards_hidden = keras.layers.Dense(128, activation='relu')(hole_cards_input)
        hole_cards_hidden = keras.layers.Dense(64, activation='relu')(hole_cards_hidden)
        
        # Process community cards
        community_cards_hidden = keras.layers.Dense(128, activation='relu')(community_cards_input)
        community_cards_hidden = keras.layers.Dense(64, activation='relu')(community_cards_hidden)
        
        # Process numeric features
        numeric_hidden = keras.layers.Dense(32, activation='relu')(numeric_input)
        numeric_hidden = keras.layers.Dense(16, activation='relu')(numeric_hidden)
        
        # Combine all features
        combined = keras.layers.Concatenate()(
            [hole_cards_hidden, community_cards_hidden, numeric_hidden]
        )
        
        # Final processing layers
        hidden = keras.layers.Dense(256, activation='relu')(combined)
        hidden = keras.layers.Dense(128, activation='relu')(hidden)
        hidden = keras.layers.Dense(64, activation='relu')(hidden)
        
        # Output layer - Q-values for each action
        output = keras.layers.Dense(self.action_size, activation='linear')(hidden)
        
        # Create and compile model
        model = keras.Model(
            inputs=[hole_cards_input, community_cards_input, numeric_input],
            outputs=output
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return model

    def preprocess_state(self, state):
        """
        Convert a poker state dictionary into the neural network's input format.
        
        Args:
            state (dict): Dictionary containing hole_cards, community_cards,
                         and numeric features
        
        Returns:
            tuple: Three numpy arrays for hole cards, community cards,
                  and numeric features
        """
        # Initialize one-hot encodings for cards
        hole_cards = np.zeros(self.hole_cards_dim)
        community_cards = np.zeros(self.community_cards_dim)
        
        # Convert hole cards to one-hot encoding
        for card in state['hole_cards']:
            idx = self._card_to_index(card)
            hole_cards[idx] = 1
            
        # Convert community cards to one-hot encoding
        for card in state['community_cards']:
            idx = self._card_to_index(card)
            community_cards[idx] = 1
            
        # Create numeric features array
        numeric_features = np.array([
            state['pot'],
            state['stack'],
            state['position'],
            state['opponent_stack'],
            state['equity']
        ])
        
        # Reshape for batch processing
        return [
            hole_cards.reshape(1, -1),
            community_cards.reshape(1, -1),
            numeric_features.reshape(1, -1)
        ]

    def _card_to_index(self, card):
        """
        Convert a card string (e.g., 'Ah') to its index in the one-hot encoding.
        
        The indexing system follows these rules:
        - Number cards 2-9: base index is card value + 1
        - Face cards have specific indices:
          T(10) -> 10 (or 34 for Ts)
          J -> 11
          Q -> 12
          K -> 13 (or 47 for Kd)
          A -> 14 (or 48 for Ah)
        
        Args:
            card (str): Two-character string representing a card (e.g., 'Ah')
        
        Returns:
            int: Index in the one-hot encoding (0-51)
        """
        rank = card[0]
        suit = card[1]
        
        # Handle special cases first
        if rank == 'T':
            base = 34 if suit == 's' else 10
        elif rank == 'J':
            base = 11
        elif rank == 'Q':
            base = 12
        elif rank == 'K':
            base = 47 if suit == 'd' else 13
        elif rank == 'A':
            base = 48 if suit == 'h' else 14
        elif rank >= '2' and rank <= '9':
            base = int(rank) + 1
        else:
            raise ValueError(f"Invalid card rank: {rank}")
        
        return base

    def predict(self, state):
        """
        Get Q-values for all actions in the given state.
        
        Args:
            state (dict): Current poker state
            
        Returns:
            numpy.ndarray: Q-values for each possible action
        """
        processed_state = self.preprocess_state(state)
        return self.model.predict(processed_state)

    def update_weights(self, state, target_q_values, episode=None, action_taken=None, reward=None):
        """
        Update the network weights using backpropagation and track performance metrics.
        
        Args:
            state (dict): Current poker state
            target_q_values (numpy.ndarray): Target Q-values for training
            episode (int, optional): Current training episode number
            action_taken (int, optional): Action that was taken in this state
            reward (float, optional): Reward received for the action
            
        Returns:
            History: Training history object
        """
        processed_state = self.preprocess_state(state)
        history = self.model.fit(processed_state, target_q_values, verbose=0)
        
        # Get current predictions for comparison
        current_q_values = self.model.predict(processed_state)
        
        # Record performance metrics with consistent types
        from datetime import datetime
        import json
        
        self.training_history['timestamp'].append(datetime.now().isoformat())
        self.training_history['episode'].append(int(episode if episode is not None else len(self.training_history['episode'])))
        self.training_history['loss'].append(float(history.history['loss'][0]))
        self.training_history['action_taken'].append(int(action_taken if action_taken is not None else -1))
        self.training_history['reward'].append(float(reward if reward is not None else 0.0))
        
        # Record relevant state information
        state_info = {
            'hand': '_'.join(state['hole_cards']),
            'position': state['position'],
            'equity': state['equity'],
            'pot': state['stack'] / state['pot'] if state['pot'] > 0 else 0  # pot odds
        }
        self.training_history['state_info'].append(json.dumps(state_info))
        
        # Record Q-value predictions
        self.training_history['q_values'].append(current_q_values.tolist())
        
        return history

    def save_performance_history(self, filename=None):
        """
        Save the model's performance history to a CSV file.
        
        Args:
            filename (str, optional): Name for the CSV file. If not provided,
                                    uses model_name and timestamp.
        
        Returns:
            Path: Path to the saved CSV file
        """
        import pandas as pd
        from datetime import datetime
        
        # Create DataFrame from history
        df = pd.DataFrame({
            'timestamp': self.training_history['timestamp'],
            'episode': self.training_history['episode'],
            'loss': self.training_history['loss'],
            'action_taken': self.training_history['action_taken'],
            'reward': self.training_history['reward'],
            'state_info': self.training_history['state_info'],
            'q_values': [str(q) for q in self.training_history['q_values']]
        })
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.model_name}_history_{timestamp}.csv"
            
        # Save to CSV in the model's directory
        save_path = self.directories['saved_models'] / self.model_name / filename
        df.to_csv(save_path, index=False)
        
        return save_path

    def save_model(self, filename):
        """
        Save the model architecture, weights, and performance history.
        
        Args:
            filename (str): Base name for the saved model files
            
        Returns:
            tuple: Paths to saved model, weights files, and performance history
        """
        save_dir = self.directories['saved_models'] / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f"{filename}.keras"
        weights_path = save_dir / f"{filename}.weights.h5"
        
        try:
            self.model.save(str(model_path))
            self.model.save_weights(str(weights_path))
            # Save performance history alongside model
            history_path = self.save_performance_history(f"{filename}_history.csv")
            return model_path, weights_path, history_path
            
        except Exception as e:
            raise RuntimeError(f"Error saving model: {str(e)}")

    def load_model(self, filename):
        """
        Load model weights from a file.
        
        Args:
            filename (str): Name of the weights file to load
            
        Returns:
            bool: True if loading was successful
        """
        weights_path = self.directories['saved_models'] / self.model_name / f"{filename}.weights.h5"
        
        try:
            self.model.load_weights(str(weights_path))
            return True
        except Exception as e:
            raise FileNotFoundError(f"Error loading weights: {str(e)}")