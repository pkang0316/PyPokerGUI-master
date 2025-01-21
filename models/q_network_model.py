import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import warnings
from datetime import datetime

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
            'loss': [],            # Training loss
            'action_taken': [],    # What action was chosen
            'reward': [],          # Reward received
            'state_info': [],      # Key state information (e.g., hand type, position)
            'q_values': []         # Model's Q-value predictions
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
    
    def _index_to_card(self, index):
        """Maps index in 52-card deck back to card string."""
        suit_map = {0: 'S', 13: 'H', 26: 'D', 39: 'C'}
        suit = suit_map[index - (index % 13)]
        
        rank_idx = index % 13
        if rank_idx >= 0 and rank_idx <= 7:
            rank = str(rank_idx + 2)
        elif rank_idx == 8:
            rank = 'T'
        elif rank_idx == 9:
            rank = 'J'
        elif rank_idx == 10:
            rank = 'Q'
        elif rank_idx == 11:
            rank = 'K'
        else:  # rank_idx == 12
            rank = 'A'
            
        return suit + rank

    def _card_to_index(self, card):
        """Maps card to index in 52-card deck one-hot encoding."""
        suit = card[0].lower()
        rank = card[1]
        
        suit_offset = {'s': 0, 'h': 13, 'd': 26, 'c': 39}
        if suit not in suit_offset:
            raise ValueError(f"Invalid card suit: {suit}")
            
        if rank == 'T':
            rank_idx = 8  # 10 is at index 8
        elif rank == 'J':
            rank_idx = 9
        elif rank == 'Q':
            rank_idx = 10
        elif rank == 'K':
            rank_idx = 11
        elif rank == 'A':
            rank_idx = 12
        elif rank >= '2' and rank <= '9':
            rank_idx = int(rank) - 2
        else:
            raise ValueError(f"Invalid card rank: {rank}")
            
        return suit_offset[suit] + rank_idx
    
    def predict(self, valid_actions, state):
        """Get Q-values for valid actions in current state."""
        hole_cards, community_cards, numeric_features = state
        
        # Decode and print cards
        hole_card_str = [self._index_to_card(i) for i, val in enumerate(hole_cards[0]) if val == 1]
        community_card_str = [self._index_to_card(i) for i, val in enumerate(community_cards[0]) if val == 1]
        print(f"Hole cards: {hole_card_str}")
        print(f"Community cards: {community_card_str}")

        # Predict Q-values
        q_values = self.model.predict([hole_cards, community_cards, numeric_features]).flatten()
        
        # Map and print Q-values for each action
        valid_q_values = np.full(len(valid_actions), -np.inf)
        print("\nAction Q-values:")
        for i, action_info in enumerate(valid_actions):
            action = action_info['action']
            amount = action_info['amount']
            if action == 'fold':
                valid_q_values[i] = q_values[0]
                print(f"FOLD: {q_values[0]:.3f}")
            elif action == 'call':
                valid_q_values[i] = q_values[1]
                print(f"CALL ({amount}): {q_values[1]:.3f}")
            elif action == 'raise':
                raise_idx = np.clip(2 + (amount['min'] / amount['max']) * (self.action_size - 3), 2, self.action_size - 1).astype(int)
                valid_q_values[i] = q_values[raise_idx]
                print(f"RAISE ({amount['min']}-{amount['max']}): {q_values[raise_idx]:.3f}")
                
        return valid_q_values

    def update_weights(self, state, target_q_values, episode=None, action_taken=None, reward=None):
        history = self.model.fit(
            [state[0], state[1], state[2]], 
            target_q_values,
            verbose=0
        )

        current_q_values = self.model.predict([state[0], state[1], state[2]])
        
        self.training_history['timestamp'].append(datetime.now().isoformat())
        self.training_history['episode'].append(int(episode if episode is not None else len(self.training_history['episode'])))
        self.training_history['loss'].append(float(history.history['loss'][0]))
        self.training_history['action_taken'].append(int(action_taken if action_taken is not None else -1))
        self.training_history['reward'].append(float(reward if reward is not None else 0.0))
        
        numeric_features = state[2][0]
        state_info = {
            'hand': state[0].tolist(),
            'position': int(numeric_features[2]),
            'equity': float(numeric_features[4]),
            'pot_odds': float(numeric_features[1] / numeric_features[0]) if numeric_features[0] > 0 else 0.0
        }
        self.training_history['state_info'].append(state_info)
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
            'state_info': [json.dumps(info) for info in self.training_history['state_info']],
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
        save_dir = self.directories['saved_models'] / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model architecture and compile it with same settings
        model_config = self.model.get_config()
        weights = self.model.get_weights()
        
        # Recreate identical model and set weights
        new_model = keras.Model.from_config(model_config)
        new_model.set_weights(weights)
        new_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        # Save complete model with weights and optimizer state
        model_path = save_dir / f"{filename}.keras"
        new_model.save(str(model_path))
        
        # Also save weights separately
        weights_path = save_dir / f"{filename}.weights.h5"
        new_model.save_weights(str(weights_path))
        
        # Save performance history
        history_path = self.save_performance_history(f"{filename}_history.csv")
        return model_path, weights_path, history_path

    def load_model(self, filename):
        try:
            model_path = self.directories['saved_models'] / self.model_name / f"{filename}.keras"
            if model_path.exists():
                self.model = keras.models.load_model(str(model_path))
            else:
                # Fallback to weights-only loading
                weights_path = self.directories['saved_models'] / self.model_name / f"{filename}.h5"
                self.model.load_weights(str(weights_path))
            
            # Load training history
            history_path = self.directories['saved_models'] / self.model_name / f"{filename}.csv"
            if history_path.exists():
                import pandas as pd
                df = pd.read_csv(history_path)
                for key in self.training_history:
                    if key in df.columns:
                        self.training_history[key] = df[key].tolist()
            
            return True
            
        except Exception as e:
            warnings.warn(f"Error loading model: {str(e)}")
            return False