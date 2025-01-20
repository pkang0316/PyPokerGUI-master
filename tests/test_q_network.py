import unittest
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil
from models.q_network_model import QNetworkModel 

class TestQNetworkModel(unittest.TestCase):
    def setUp(self):
        """Set up test environment with sample data and model instance."""
        # Initialize model with 3 possible actions (fold, call, raise)
        self.action_size = 3
        self.model = QNetworkModel(action_size=self.action_size)
        
        # Create a sample poker state for testing
        self.valid_state = {
            'hole_cards': ['Ah', 'Kh'],
            'community_cards': ['7s', '8s', '9s'],
            'pot': 100,
            'stack': 1000,
            'position': 2,
            'opponent_stack': 900,
            'equity': 0.75
        }
        
        # Set up temporary directory for model artifacts
        self.test_dir = Path('test_models')
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test artifacts after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Verify model initialization and architecture."""
        # Test model dimensions
        self.assertEqual(self.model.action_size, self.action_size)
        self.assertEqual(self.model.hole_cards_dim, 52)
        self.assertEqual(self.model.community_cards_dim, 52)
        self.assertEqual(self.model.numeric_features_dim, 5)
        
        # Test neural network structure
        self.assertEqual(len(self.model.model.inputs), 3)
        self.assertEqual(len(self.model.model.outputs), 1)
        self.assertEqual(self.model.model.outputs[0].shape[-1], self.action_size)

    def test_card_to_index(self):
        """Verify card index conversion for various cards."""
        test_cases = [
            ('Ah', 48),  # Ace of Hearts
            ('Kd', 47),  # King of Diamonds
            ('2c', 3),   # Two of Clubs
            ('Ts', 34),  # Ten of Spades
        ]
        
        for card, expected_index in test_cases:
            with self.subTest(card=card):
                self.assertEqual(
                    self.model._card_to_index(card), 
                    expected_index,
                    f"Failed to convert {card} to correct index"
                )

    def test_preprocess_state(self):
        """Verify state preprocessing functionality."""
        processed = self.model.preprocess_state(self.valid_state)
        
        # Check output structure
        self.assertEqual(len(processed), 3)
        self.assertEqual(processed[0].shape, (1, 52))  # hole cards
        self.assertEqual(processed[1].shape, (1, 52))  # community cards
        self.assertEqual(processed[2].shape, (1, 5))   # numeric features
        
        # Verify one-hot encoding of cards
        hole_cards = processed[0][0]
        ah_index = self.model._card_to_index('Ah')
        kh_index = self.model._card_to_index('Kh')
        self.assertEqual(hole_cards[ah_index], 1)
        self.assertEqual(hole_cards[kh_index], 1)
        self.assertEqual(sum(hole_cards), 2)

    def test_predict(self):
        """Verify model prediction functionality."""
        prediction = self.model.predict(self.valid_state)
        
        # Check prediction shape and numerical validity
        self.assertEqual(prediction.shape, (1, self.action_size))
        self.assertTrue(np.all(np.isfinite(prediction)))

    def test_model_training(self):
        """
        Verify that the model can process training data and update weights.
        This test focuses on the mechanics of training rather than the quality
        of decisions.
        """
        # Create a batch of training data
        training_states = []
        target_q_values = []
        
        # Generate 5 different poker states with varied features
        for i in range(5):
            state = {
                'hole_cards': ['Ah', 'Kh'],
                'community_cards': ['7s', '8s', '9s'],
                'pot': 100 * (i + 1),  # Vary pot sizes
                'stack': 1000 - (i * 100),  # Vary stack sizes
                'position': i % 2,  # Alternate positions
                'opponent_stack': 900 + (i * 50),  # Vary opponent stacks
                'equity': 0.75 - (i * 0.1)  # Vary equity
            }
            training_states.append(state)
            
            # Create target Q-values (the exact values don't matter,
            # we just want to verify the model can train on them)
            target = np.zeros((1, self.action_size))
            target[0, i % self.action_size] = 1.0
            target_q_values.append(target)
        
        # Train model for a few iterations
        histories = []
        for state, target in zip(training_states, target_q_values):
            history = self.model.update_weights(state, target)
            histories.append(history)
            
            # Verify training mechanics
            self.assertIsNotNone(history)
            self.assertTrue('loss' in history.history)
            self.assertTrue(len(history.history['loss']) > 0)
            
            # Verify loss is a valid number
            self.assertTrue(np.isfinite(history.history['loss'][0]))

    def test_save_load_model(self):
        """Verify model persistence operations."""
        test_filename = "test_model"
        
        # Get predictions before saving
        initial_prediction = self.model.predict(self.valid_state)
        
        # Test save functionality
        model_path, weights_path, history_path = self.model.save_model(test_filename)
        self.assertTrue(Path(model_path).exists())
        self.assertTrue(Path(weights_path).exists())
        self.assertTrue(Path(history_path).exists())
        
        # Test load functionality
        new_model = QNetworkModel(action_size=self.action_size)
        new_model.load_model(test_filename)
        
        # Verify loaded model produces same predictions
        loaded_prediction = new_model.predict(self.valid_state)
        np.testing.assert_array_almost_equal(
            initial_prediction,
            loaded_prediction,
            decimal=5,
            err_msg="Loaded model predictions don't match original"
        )

    def test_extended_training_behavior(self):
        """
        Explore the model's training behavior with realistic poker scenarios.
        This test examines how the model adapts to different poker situations,
        but does not enforce specific decision requirements.
        
        Note: This test explores learning patterns but doesn't fail if the model
        makes different decisions - it only verifies that meaningful changes occur
        during training.
        """
        # Create diverse poker scenarios that represent common situations
        training_states = [
            {
                'hole_cards': ['Ah', 'Kh'],  # Premium hand
                'community_cards': ['Qh', 'Jh', 'Th'],  # Royal flush draw
                'pot': 100,
                'stack': 1000,
                'position': 1,  # In position
                'opponent_stack': 900,
                'equity': 0.85
            },
            {
                'hole_cards': ['2c', '7d'],  # Weak hand
                'community_cards': ['As', 'Ks', 'Qs'],  # Strong board
                'pot': 200,
                'stack': 500,
                'position': 0,  # Out of position
                'opponent_stack': 1500,
                'equity': 0.15
            },
            {
                'hole_cards': ['Ts', 'Js'],  # Drawing hand
                'community_cards': ['9s', '8s', '2h'],  # Flush draw
                'pot': 150,
                'stack': 750,
                'position': 1,
                'opponent_stack': 600,
                'equity': 0.45
            }
        ]

        # Set up target Q-values that represent reasonable actions
        # Format: [fold, call, raise]
        target_q_values = [
            [[0.1, 0.4, 0.9]],  # Premium hand scenario
            [[0.8, 0.3, 0.1]],  # Weak hand scenario
            [[0.2, 0.7, 0.4]]   # Drawing hand scenario
        ]

        # Collect initial predictions for comparison
        initial_predictions = []
        for state in training_states:
            pred = self.model.predict(state)
            initial_predictions.append(pred)

        # Train the model multiple times on each scenario
        n_epochs = 50
        training_history = []
        
        for epoch in range(n_epochs):
            epoch_history = []
            for state, target in zip(training_states, target_q_values):
                history = self.model.update_weights(state, np.array(target))
                epoch_history.append(history.history['loss'][0])
            training_history.append(np.mean(epoch_history))
            
            # Optional early stopping for efficiency
            if np.mean(epoch_history) < 0.01:
                break

        # Collect final predictions after training
        final_predictions = []
        for state in training_states:
            pred = self.model.predict(state)
            final_predictions.append(pred)

        # Verify that training produced meaningful changes
        for i in range(len(training_states)):
            # Calculate the magnitude of change in predictions
            prediction_change = np.sum(np.abs(final_predictions[i] - initial_predictions[i]))
            
            # Verify that predictions changed during training
            self.assertGreater(
                prediction_change,
                0.0,
                f"Training did not affect predictions for state {i}"
            )

        # Verify that loss generally decreased during training
        # We look at trend rather than absolute values
        early_loss_avg = np.mean(training_history[:5])
        late_loss_avg = np.mean(training_history[-5:])
        self.assertLess(
            late_loss_avg,
            early_loss_avg,
            "Training loss did not show a decreasing trend"
        )

    def test_model_adaptability(self):
        """
        Test the model's ability to adapt to changing poker situations.
        This test verifies that the model can modify its behavior based on
        training, without requiring specific decisions.
        """
        # Create a sequence of related states with varying stack sizes
        base_state = self.valid_state.copy()
        stack_sizes = [100, 500, 1000, 2000, 5000]
        
        states = []
        for stack in stack_sizes:
            state = base_state.copy()
            state['stack'] = stack
            state['opponent_stack'] = stack
            states.append(state)

        # Train the model on each state sequentially
        for state in states:
            # Create target that encourages different actions based on stack size
            target = np.zeros((1, self.action_size))
            if state['stack'] < 1000:
                target[0] = [0.5, 0.3, 0.2]  # Conservative with small stacks
            else:
                target[0] = [0.2, 0.3, 0.5]  # Aggressive with big stacks
                
            # Train multiple times on each state
            for _ in range(10):
                history = self.model.update_weights(state, target)
                
                # Verify training operation worked
                self.assertIsNotNone(history)
                self.assertTrue('loss' in history.history)
                self.assertTrue(np.isfinite(history.history['loss'][0]))

        # Verify predictions are different for small vs large stacks
        small_stack_pred = self.model.predict(states[0])
        large_stack_pred = self.model.predict(states[-1])
        
        # Check that predictions changed based on stack size
        self.assertTrue(
            np.any(np.abs(small_stack_pred - large_stack_pred) > 0.1),
            "Model does not show sensitivity to stack size changes"
        )

    def test_performance_tracking(self):
        """
        Test that the model correctly tracks performance metrics during training.
        This test verifies that our performance tracking system captures and stores
        all the necessary information about the model's learning process.
        """
        # Train model several times to generate performance data
        episodes = 5
        actions = [0, 1, 2]  # fold, call, raise
        rewards = [10, -5, 20]
        
        for episode in range(episodes):
            # Create slightly different states for each episode
            state = self.valid_state.copy()
            state['pot'] = 100 * (episode + 1)  # Vary the pot size
            state['stack'] = 1000 - (episode * 100)  # Vary the stack size
            
            # Target Q-values that prefer different actions
            target = np.zeros((1, self.action_size))
            target[0, episode % self.action_size] = 1.0  # Rotate preferred action
            
            # Update weights and track performance
            self.model.update_weights(
                state=state,
                target_q_values=target,
                episode=episode,
                action_taken=actions[episode % len(actions)],
                reward=rewards[episode % len(rewards)]
            )
        
        # Verify that performance data was collected correctly
        history = self.model.training_history
        
        # Check that we have the expected number of data points
        self.assertEqual(len(history['episode']), episodes)
        self.assertEqual(len(history['loss']), episodes)
        self.assertEqual(len(history['action_taken']), episodes)
        self.assertEqual(len(history['reward']), episodes)
        self.assertEqual(len(history['q_values']), episodes)
        
        # Verify data types and ranges
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in history['episode']))
        self.assertTrue(all(isinstance(x, (float, np.floating)) for x in history['loss']))
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in history['action_taken']))
        self.assertTrue(all(isinstance(x, (float, np.floating)) for x in history['reward']))
        
        # Verify episodes are sequential
        self.assertEqual(history['episode'], list(range(episodes)))
        
        # Verify actions are within valid range
        self.assertTrue(all(0 <= a <= 2 for a in history['action_taken']))
        
        # Verify Q-values are properly shaped
        self.assertTrue(all(len(q[0]) == self.action_size for q in history['q_values']))

    def test_save_performance_history(self):
        """
        Test that performance history can be properly saved to a CSV file.
        This test verifies that our performance data is correctly persisted
        and can be loaded back for analysis.
        """
        import pandas as pd
        
        # Generate some performance data
        state = self.valid_state
        target = np.array([[0.8, 0.1, 0.1]])  # Prefer folding
        
        # Perform a few training updates
        for i in range(3):
            self.model.update_weights(
                state=state,
                target_q_values=target,
                episode=i,
                action_taken=0,  # fold
                reward=10.0
            )
        
        # Save performance history
        test_filename = "test_performance.csv"
        history_path = self.model.save_performance_history(test_filename)
        
        # Verify file exists
        self.assertTrue(Path(history_path).exists())
        
        # Load and verify CSV content
        df = pd.read_csv(history_path)
        
        # Check that all expected columns are present
        expected_columns = [
            'timestamp', 'episode', 'loss', 'action_taken',
            'reward', 'state_info', 'q_values'
        ]
        self.assertTrue(all(col in df.columns for col in expected_columns))
        
        # Verify row count matches training updates
        self.assertEqual(len(df), 3)
        
        # Verify numerical columns have correct values
        self.assertTrue(all(df['episode'] == [0, 1, 2]))
        self.assertTrue(all(df['action_taken'] == 0))
        self.assertTrue(all(df['reward'] == 10.0))
        
        # Verify state info is properly serialized
        import json
        state_info = json.loads(df['state_info'].iloc[0])
        self.assertEqual(state_info['hand'], 'Ah_Kh')
        self.assertEqual(state_info['position'], 2)
        
    def test_saving_with_history(self):
        """
        Test that model saving now includes performance history.
        This test verifies that when we save a model, its performance
        history is properly saved alongside the model and weights.
        """
        test_filename = "test_model_with_history"
        
        # Generate some performance data
        state = self.valid_state
        target = np.array([[0.8, 0.1, 0.1]])
        self.model.update_weights(state, target, episode=0, action_taken=0, reward=10.0)
        
        # Save model with history
        model_path, weights_path, history_path = self.model.save_model(test_filename)
        
        # Verify all files exist
        self.assertTrue(Path(model_path).exists())
        self.assertTrue(Path(weights_path).exists())
        self.assertTrue(Path(history_path).exists())