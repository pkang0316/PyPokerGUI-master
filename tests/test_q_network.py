import unittest
import numpy as np
from models.q_network_model import QNetworkModel
from players.q_network_model_player_setup import QLearningPokerPlayer

class TestQNetworkModel(unittest.TestCase):
    
    def setUp(self):
        self.model = QNetworkModel(action_size=6)
        self.player = QLearningPokerPlayer("test_player")
        self.player.model = self.model
        
    def test_preprocess_state(self):
        """
        Test that the preprocess_state method correctly converts game state 
        into the format expected by the model.
        """
        hole_card = ['Ah', 'Kd'] 
        round_state = {
            'community_card': ['2s', '3h', 'Jc'],
            'pot': {'main': {'amount': 100}},
            'seats': [
                {'uuid': '1', 'stack': 950},
                {'uuid': '2', 'stack': 1050}
            ]
        }
        
        # Set player's uuid to match a seat
        self.player.uuid = '1'
        
        processed_state = self.player.preprocess_state(hole_card, round_state)
        
        # Check that the output has the expected shape
        self.assertEqual(len(processed_state), 3)
        self.assertEqual(processed_state[0].shape, (1, 52))  # hole cards
        self.assertEqual(processed_state[1].shape, (1, 52))  # community cards   
        self.assertEqual(processed_state[2].shape, (1, 5))   # numeric features
        
        # Check that the hole cards are correctly encoded
        self.assertEqual(processed_state[0][0][48], 1)  # Ah
        self.assertEqual(processed_state[0][0][47], 1)  # Kd
        
        # Check that the community cards are correctly encoded  
        self.assertEqual(processed_state[1][0][3], 1)   # 2s
        self.assertEqual(processed_state[1][0][4], 1)   # 3h
        self.assertEqual(processed_state[1][0][11], 1)  # Jc
        
        # Check that the numeric features are in the expected ranges
        self.assertEqual(processed_state[2][0][0], 0.1)                 # pot size (100 / 1000)
        self.assertEqual(processed_state[2][0][1], 0.95)                # stack (950 / 1000) 
        self.assertEqual(processed_state[2][0][2], 0)                   # position (0-indexed)
        self.assertEqual(processed_state[2][0][3], 1.05)                # opp stack (1050 / 1000)
        self.assertTrue(0 <= processed_state[2][0][4] <= 1)            # equity (placeholder for now)
        
    def test_declare_action(self):
        """
        Test that the declare_action method correctly selects an action based
        on the predicted Q-values.
        """
        valid_actions = [
            {'action': 'fold', 'amount': 0},
            {'action': 'call', 'amount': 10},
            {'action': 'raise', 'amount': {'min': 20, 'max': 100}}
        ]
        hole_card = ['Ah', 'Kd']
        round_state = {
            'community_card': ['2s', '3h', 'Jc'],
            'pot': {'main': {'amount': 100}},
            'seats': [
                {'uuid': '1', 'stack': 950},
                {'uuid': '2', 'stack': 1050}  
            ]
        }
        
        # Set player's uuid to match a seat
        self.player.uuid = '1'
        
        # Set up the model to always predict the same Q-values
        q_values = np.array([0.2, 0.5, 0.1, 0.1, 0.1, 0.1])
        self.player.model.predict = lambda _, __: q_values
        
        # Test that it selects the action with the highest Q-value
        action, amount = self.player.declare_action(valid_actions, hole_card, round_state)
        self.assertEqual(action, 'call')
        self.assertEqual(amount, 10)
        
        # Change the Q-values and test again
        q_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        action, amount = self.player.declare_action(valid_actions, hole_card, round_state)
        self.assertEqual(action, 'raise')
        self.assertEqual(amount['min'], 20)
        self.assertEqual(amount['max'], 100)
        
    def test_model_prediction(self):
        """
        Test that the model's predict method correctly maps its Q-value outputs
        to the given valid actions.
        """
        valid_actions = [
            {'action': 'fold', 'amount': 0},
            {'action': 'call', 'amount': 10},
            {'action': 'raise', 'amount': {'min': 20, 'max': 100}}
        ]
        
        # Create a game state
        state = (
            np.ones((1, 52)),  # hole cards 
            np.ones((1, 52)),  # community cards
            np.array([[0.1, 0.2, 0, 0.3, 0.4]])  # numeric features
        )
        
        # Set up the model to always predict the same Q-values 
        q_values = np.array([0.2, 0.5, 0.1, 0.1, 0.1, 0.1])
        self.model.model.predict = lambda _: q_values
        
        # Get the masked Q-values for the valid actions
        masked_q_values = self.model.predict(valid_actions, state)
        
        # Check that the Q-values are correctly mapped
        self.assertEqual(masked_q_values[0], 0.2)  # fold
        self.assertEqual(masked_q_values[1], 0.5)  # call
        self.assertEqual(masked_q_values[2], 0.1)  # raise (min)
        
    def test_model_update(self):
        """
        Test that the model's update_weights method correctly updates the
        model weights and records training metrics.
        """
        # Create a game state and target Q-values
        state = (
            np.ones((1, 52)),  # hole cards
            np.ones((1, 52)),  # community cards  
            np.array([[0.1, 0.2, 0, 0.3, 0.4]])  # numeric features
        )
        target_q_values = np.array([[0.2, 0.5, 0.1, 0.1, 0.1, 0.1]])
        
        # Update the model weights
        history = self.model.update_weights(state, target_q_values, episode=1, action_taken=1, reward=10)
        
        # Check that the weights were updated
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertGreater(len(history.history['loss']), 0)
        
        # Check that the training metrics were recorded
        self.assertEqual(len(self.model.training_history['episode']), 1)
        self.assertEqual(self.model.training_history['episode'][0], 1)
        self.assertEqual(self.model.training_history['action_taken'][0], 1)
        self.assertEqual(self.model.training_history['reward'][0], 10)
        
        # Update the weights again and check that the metrics were appended
        history = self.model.update_weights(state, target_q_values, episode=2, action_taken=2, reward=20)
        self.assertEqual(len(self.model.training_history['episode']), 2)
        self.assertEqual(self.model.training_history['episode'][1], 2)
        self.assertEqual(self.model.training_history['action_taken'][1], 2)  
        self.assertEqual(self.model.training_history['reward'][1], 20)
        
    def test_model_save_load(self):
        """
        Test that the model can be correctly saved to and loaded from files,
        including its weights and training history.
        """
        # Update the model weights to create some history
        state = (
            np.ones((1, 52)), 
            np.ones((1, 52)),
            np.array([[0.1, 0.2, 0, 0.3, 0.4]])
        )
        target_q_values = np.array([[0.2, 0.5, 0.1, 0.1, 0.1, 0.1]])
        self.model.update_weights(state, target_q_values, episode=1, action_taken=1, reward=10)
        
        # Save the model
        model_path, weights_path, history_path = self.model.save_model('test_model')
        self.assertTrue(model_path.exists())
        self.assertTrue(weights_path.exists()) 
        self.assertTrue(history_path.exists())
        
        # Load the model into a new instance  
        loaded_model = QNetworkModel(action_size=6)
        loaded_model.load_model('test_model')
        
        # Check that the loaded weights give the same prediction
        q_values = loaded_model.model.predict(state)
        original_q_values = self.model.model.predict(state)
        np.testing.assert_array_almost_equal(q_values, original_q_values)
        
        # Check that the loaded history matches the original
        loaded_history = loaded_model.training_history
        self.assertEqual(loaded_history['episode'], self.model.training_history['episode'])
        self.assertEqual(loaded_history['action_taken'], self.model.training_history['action_taken'])
        self.assertEqual(loaded_history['reward'], self.model.training_history['reward'])

if __name__ == '__main__':
    unittest.main()