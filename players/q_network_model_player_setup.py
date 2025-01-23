import numpy as np
from pypokerengine.players import BasePokerPlayer
from models.q_network_model import QNetworkModel 
from pathlib import Path
import uuid

ACTION_SIZE = 6

class QLearningPokerPlayer(BasePokerPlayer):
    def __init__(self, player_name):
        super().__init__()
        self.model = None
        self.player_name = player_name
        self.uuid = None
        self.episode_counter = 0
        self.last_state = None
        self.last_action = None
    
    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.preprocess_state(hole_card, round_state)
        self.last_state = state
        
        q_values = self.model.predict(valid_actions, state)
        
        # Apply softmax to convert Q-values to probabilities
        probabilities = np.exp(q_values[:len(valid_actions)]) / np.sum(np.exp(q_values[:len(valid_actions)]))
        
        # Sample action based on probabilities
        action_idx = np.random.choice(len(valid_actions), p=probabilities)
        
        self.last_action = action_idx
        action = valid_actions[action_idx]['action']
        amount = valid_actions[action_idx]['amount']
        
        if isinstance(amount, dict):
            amount = np.mean(list(amount.values()))
        return action, amount

    def receive_game_start_message(self, game_info):
        """Initialize model and UUID for new game."""
        self.model = QNetworkModel(6)
        self.model.load_model(f'{self.player_name}.weights')
        
        # Get UUID from game info if available, otherwise generate one
        self.uuid = game_info.get('uuid', str(uuid.uuid4()))

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def _calculate_reward(self, round_state):
        """Calculate reward based on action outcome."""
        if not hasattr(self, 'initial_stack'):
            self.initial_stack = self._get_player_stack(round_state)
            return 0
            
        current_stack = self._get_player_stack(round_state)
        print(f"current_stack:{current_stack}")
        reward = current_stack - self.initial_stack
        print(f"initial_stack:{self.initial_stack}")
        self.initial_stack = current_stack
        return reward
    
    def _get_player_stack(self, round_state):
        """Get player's current stack size."""
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                return seat['stack']
        return 0
    
    def receive_game_update_message(self, action, round_state):
        if not (self.last_state and self.last_action is not None):
            return
            
        try:
            reward = self._calculate_reward(round_state)
            
            # Create full action space target values
            target_q_values = np.zeros((1, 6))  # Match model output shape
            target_q_values[0, self.last_action] = reward
            
            self.episode_counter += 1
            self.model.update_weights(
                self.last_state,
                target_q_values,
                episode=self.episode_counter,
                action_taken=self.last_action,
                reward=reward
            )
            print(f"Updated weights - Episode {self.episode_counter}, Reward {reward}")
        except Exception as e:
            print(f"Failed to update model weights: {str(e)}")
            print(f"Debug - last_action: {self.last_action}, reward: {reward}")

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Update weights one final time with game outcome
        print(f"round_state:{round_state}")
        print(f"winners:{winners}")
        print(f"hand_info:{hand_info}")
        if self.last_state and self.last_action is not None:
            try:
                reward = self._calculate_reward(round_state)
                
                target_q_values = np.zeros((1, 6))
                target_q_values[0, self.last_action] = reward
                
                self.model.update_weights(
                    self.last_state,
                    target_q_values,
                    episode=self.episode_counter,
                    action_taken=self.last_action,
                    reward=reward
                )
                print(f"Final update - Episode {self.episode_counter}, Reward {reward}")
                
            except Exception as e:
                print(f"Failed to update final weights: {str(e)}")
        
        # Save model after weights update
        if self.model:
            self.model.save_model(self.player_name)

    def preprocess_state(self, hole_card, round_state):
        """
        Convert the current poker game state into the neural network's input format.
        
        Args:
            hole_card (list): The player's private hole cards
            round_state (dict): The current round state dictionary
        
        Returns:
            tuple: Three numpy arrays representing the preprocessed state:
                - hole_cards: One-hot encoded hole cards 
                - community_cards: One-hot encoded community cards
                - numeric_features: Normalized numeric features
        """
        # Initialize the one-hot encoded hole cards array
        hole_cards = np.zeros(self.model.hole_cards_dim)
        
        # Convert the hole cards to one-hot encoding
        for card in hole_card:
            idx = self.model._card_to_index(card)
            hole_cards[idx] = 1
            
        # Initialize the one-hot encoded community cards array
        community_cards = np.zeros(self.model.community_cards_dim)
        
        # Convert the community cards to one-hot encoding
        for card in round_state['community_card']:
            idx = self.model._card_to_index(card)
            community_cards[idx] = 1
            
        # Get the necessary numeric features from the round state
        pot = round_state['pot']['main']['amount']
        
        # Make sure player's uuid exists in round_state before accessing stack info
        player_seat = next((seat for seat in round_state['seats'] if seat['uuid'] == self.uuid), None)
        if player_seat is None:
            raise ValueError(f"Player's UUID {self.uuid} not found in round_state")
        stack = player_seat['stack']

        position = [i for i, seat in enumerate(round_state['seats']) if seat['uuid'] == self.uuid][0]
        opponent_stack = [seat['stack'] for seat in round_state['seats'] if seat['uuid'] != self.uuid][0]
        
        # TODO: Calculate the player's equity based on hole cards and community cards
        equity = 0.5  # Placeholder value for now
        
        # Create the numeric features array
        numeric_features = np.array([
            pot,
            stack,
            position,
            opponent_stack, 
            equity
        ])
        
        # Normalize the numeric features to be in the range [0, 1]
        numeric_features[0] /= 1000  # Assume a maximum pot size of 1000
        numeric_features[1] /= 1000  # Assume a maximum stack size of 1000
        numeric_features[3] /= 1000  # Assume a maximum opponent stack size of 1000
        
        # Return the preprocessed state
        return hole_cards.reshape(1, -1), community_cards.reshape(1, -1), numeric_features.reshape(1, -1)
    
def setup_ai():
    return QLearningPokerPlayer('q_network_model')