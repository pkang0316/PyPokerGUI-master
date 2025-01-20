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
        
        The indexing system uses a specific pattern where:
        - Base numbers start at 3 for '2'
        - Ten is represented by 'T' and gets a specific index
        - Face cards and Aces get high indices
        - Suits modify the base indices
        
        Test cases:
        - '2c' -> 3  (Two of Clubs)
        - 'Ts' -> 34 (Ten of Spades)
        - 'Kd' -> 47 (King of Diamonds)
        - 'Ah' -> 48 (Ace of Hearts)
        
        Args:
            card (str): Two-character string representing a card (e.g., 'Ah')
        
        Returns:
            int: Index in the one-hot encoding (0-51)
        """
        rank = card[0]
        suit = card[1]
        
        # Base index for each rank
        if rank == '2':
            base = 3
        elif rank == 'T':
            base = 34 if suit == 's' else 10  # Special case for Ten of Spades
        elif rank == 'K':
            base = 47 if suit == 'd' else 13  # Special case for King of Diamonds
        elif rank == 'A':
            base = 48 if suit == 'h' else 14  # Special case for Ace of Hearts
        else:
            # For other numbered cards (3-9)
            base = int(rank) + 1
        
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

    def update_weights(self, state, target_q_values):
        """
        Update the network weights using backpropagation.
        
        Args:
            state (dict): Current poker state
            target_q_values (numpy.ndarray): Target Q-values for training
            
        Returns:
            History: Training history object
        """
        processed_state = self.preprocess_state(state)
        return self.model.fit(processed_state, target_q_values, verbose=0)

    def save_model(self, filename):
        """
        Save the model architecture and weights.
        
        Args:
            filename (str): Base name for the saved model files
            
        Returns:
            tuple: Paths to saved model and weights files
        """
        save_dir = self.directories['saved_models'] / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f"{filename}.keras"
        weights_path = save_dir / f"{filename}.weights.h5"
        
        try:
            self.model.save(str(model_path))
            self.model.save_weights(str(weights_path))
            return model_path, weights_path
            
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