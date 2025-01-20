import os
import time
import glob
import numpy as np
from pypokerengine.players import BasePokerPlayer
from players.neural_network.q_network.q_network_model import QNetworkModel

class QLearningPokerPlayer(BasePokerPlayer):
    def __init__(self, player_name):
        super().__init__()
        self.player_name = player_name  # Player-specific name
        self.state_size = 10  # This should match the size of your feature vector
        self.action_size = 3  # Assuming actions are fold, call, raise (can be adjusted)
        self.q_network = QNetworkModel(self.state_size, self.action_size)
        self.epsilon = 0.1  # Exploration rate (for epsilon-greedy policy)
        self.gamma = 0.95  # Discount factor for future rewards
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self.total_rewards = 0  # To track accumulated reward over time for evaluation
        self.load_model_if_exists()

    def load_model_if_exists(self):
        # Find all files matching the pattern for the player with timestamp
        search_pattern = f"q_network_model_{self.player_name}_*.weights.h5"
        
        print(f"Looking for weight files: {search_pattern}")
        model_files = glob.glob(search_pattern)
        print(f"Found weight files: {model_files}")
        
        if model_files:
            # Sort the files based on the timestamp in the filename (descending order)
            model_files.sort(reverse=True)  # This sorts from the latest to the oldest
            latest_model_filename = model_files[0]
            print(f"Loading latest weights for {self.player_name} from {latest_model_filename}...")
            self.q_network.load_model(latest_model_filename)
        else:
            print(f"No saved weights found for {self.player_name}, starting fresh.")
        
    def declare_action(self, valid_actions, hole_card, round_state):
        # Process the game state and create a state vector
        state = self.process_state(round_state)
        
        # Use epsilon-greedy approach to choose an action
        if np.random.rand() < self.epsilon:
            # Exploration: Random action
            action = np.random.choice(valid_actions)
        else:
            # Exploitation: Use the Q-network to predict the best action
            q_values = self.q_network.predict(state)
            best_action_idx = np.argmax(q_values)
            
            # Make sure we don't exceed the index of valid actions
            best_action_idx = min(best_action_idx, len(valid_actions) - 1)
            action = valid_actions[best_action_idx]

        # Extract the correct action format
        action_info = {
            'action': action['action'],
            'amount': action.get('amount', 0)  # Default to 0 if amount not specified
        }

        # Save the last state-action pair for later use in reward collection
        self.last_state = state
        self.last_action = action_info
        
        # Return action name and amount separately
        return action_info['action'], action_info['amount']
    
    def process_state(self, round_state):
        state = np.zeros(self.state_size)
        
        try:
            # Extract relevant information from round_state
            state[0] = round_state['pot']['main']['amount'] / 1000.0  # Normalize pot size
            state[1] = len(round_state['community_card']) / 5.0  # Progress of the hand
            
            # Find our seat and stack
            our_seat = next((seat for seat in round_state['seats'] if seat['name'] == self.player_name), None)
            if our_seat:
                state[2] = our_seat['stack'] / 1000.0  # Normalize our stack
            
            # Opponent's stack
            opponent_seat = next((seat for seat in round_state['seats'] if seat['name'] != self.player_name), None)
            if opponent_seat:
                state[3] = opponent_seat['stack'] / 1000.0  # Normalize opponent stack
            
            # Position (are we dealer?)
            state[4] = 1.0 if round_state['dealer_btn'] == our_seat['uuid'] else 0.0
            
            # Last action history if available
            if round_state['action_histories']:
                last_street = list(round_state['action_histories'].keys())[-1]
                if round_state['action_histories'][last_street]:
                    last_action = round_state['action_histories'][last_street][-1]
                    state[5] = 1.0 if last_action['action'] == 'RAISE' else 0.0
                    state[6] = 1.0 if last_action['action'] == 'CALL' else 0.0
                    state[7] = 1.0 if last_action['action'] == 'FOLD' else 0.0
                    
            # Additional features can be added up to state[9] since state_size is 10
                    
        except Exception as e:
            print(f"Error processing state: {e}")
        
        return state
    
    def receive_game_start_message(self, game_info):
        print(game_info)
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        print(round_count)
        print(hole_card)
        print(seats)
        pass

    def receive_street_start_message(self, street, round_state):
        print(street)
        print(round_state)
        pass

    def receive_game_update_message(self, action, round_state):
        print(action)
        print(round_state)
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Print the entire round_state to inspect its contents (for debugging)
        print(f"Received round state: {round_state}")  # Debugging line

        action_idx = None  # Initialize action_idx to None to ensure it's always defined

        try:
            # Check if 'valid_actions' exists in round_state before proceeding
            if 'valid_actions' in round_state:
                action_idx = self.get_action_index(self.last_action, round_state['valid_actions'])
            else:
                print("Warning: 'valid_actions' not found in round_state, skipping action update.")
        except KeyError:
            # Handle the case when 'valid_actions' is missing
            print("Error: 'valid_actions' not found in round_state")
            return  # Gracefully exit the function if valid_actions is missing

        # Calculate reward based on the final stack sizes at the end of the round
        reward = self.get_reward(winners, hand_info)
        self.total_rewards += reward  # Track the total reward

        # Update the Q-network with the feedback based on the reward
        target_q_values = self.q_network.predict(self.last_state)

        # If action_idx was found (i.e., valid actions were available), update the Q-value
        if action_idx is not None:
            target_q_values[0][action_idx] = reward + self.gamma * np.max(target_q_values)  # Temporal difference
        else:
            print("No valid action found; using final stack changes to update weights.")  # Optional debug message

        # Update model weights based on the new Q-values
        self.q_network.update_weights(self.last_state, target_q_values)

        # Save the model after each round ends
        timestamp = time.strftime("%Y%m%d_%H")
        model_filename = f"q_network_model_{self.player_name}_{timestamp}.h5"
        self.save_model(model_filename)

        # Reset for the next round
        self.last_state = None
        self.last_action = None
        self.last_reward = reward

    def get_reward(self, winners, hand_info):
        # Reward function can be defined based on whether the player won or lost
        if self.is_winner(winners, hand_info):
            return 1  # Positive reward for winning
        else:
            return -1  # Negative reward for losing
    
    def is_winner(self, winners, hand_info):
        """
        Check if this player is among the winners.
        We need to ensure that 'hand_info' is in the correct structure (dict).
        """
        if isinstance(hand_info, list):
            print(f"hand_info is unexpectedly a list: {hand_info}")
            return False  # No winner if hand_info is an empty list or not structured correctly

        # Ensure 'player_name' exists in hand_info and is in the winners list
        try:
            return True if hand_info.get('player_name') in [winner['name'] for winner in winners] else False
        except KeyError as e:
            print(f"KeyError: {e} - Missing 'player_name' key in hand_info.")
            return False

    def get_action_index(self, action, valid_actions):
        # This function converts the action back into an index based on the valid actions
        for idx, valid_action in enumerate(valid_actions):
            if valid_action['action'] == action['action']:
                return idx
        return -1

    def save_model(self, filename):
        print("saving model")
        self.q_network.save_model(filename)

def setup_ai():
    return QLearningPokerPlayer('player_1')