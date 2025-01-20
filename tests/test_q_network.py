import unittest
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil
from models.q_network_model import QNetworkModel 

class TestQNetworkModel(unittest.TestCase):
    def setUp(self):
        """Set up test environment with sample data and model instance."""
        self.action_size = 3  # fold, call, raise
        self.model = QNetworkModel(action_size=self.action_size)
        
        # Sample valid state for testing
        self.valid_state = {
            'hole_cards': ['Ah', 'Kh'],
            'community_cards': ['7s', '8s', '9s'],
            'pot': 100,
            'stack': 1000,
            'position': 2,
            'opponent_stack': 900,
            'equity': 0.75
        }
        
        # Create a temporary directory for test artifacts
        self.test_dir = Path('test_models')
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test artifacts."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test model initialization and architecture."""
        self.assertEqual(self.model.action_size, self.action_size)
        self.assertEqual(self.model.hole_cards_dim, 52)
        self.assertEqual(self.model.community_cards_dim, 52)
        self.assertEqual(self.model.numeric_features_dim, 5)
        
        # Test model architecture
        self.assertEqual(len(self.model.model.inputs), 3)
        self.assertEqual(len(self.model.model.outputs), 1)
        self.assertEqual(self.model.model.outputs[0].shape[-1], self.action_size)

    def test_card_to_index(self):
        """Test card index conversion for various cards."""
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
        """Test state preprocessing with various inputs."""
        processed = self.model.preprocess_state(self.valid_state)
        
        # Test output structure
        self.assertEqual(len(processed), 3)
        self.assertEqual(processed[0].shape, (1, 52))  # hole cards
        self.assertEqual(processed[1].shape, (1, 52))  # community cards
        self.assertEqual(processed[2].shape, (1, 5))   # numeric features
        
        # Test one-hot encoding
        hole_cards = processed[0][0]
        ah_index = self.model._card_to_index('Ah')
        kh_index = self.model._card_to_index('Kh')
        self.assertEqual(hole_cards[ah_index], 1)
        self.assertEqual(hole_cards[kh_index], 1)
        self.assertEqual(sum(hole_cards), 2)

    def test_predict(self):
        """Test model prediction functionality."""
        prediction = self.model.predict(self.valid_state)
        
        # Test prediction shape and values
        self.assertEqual(prediction.shape, (1, self.action_size))
        self.assertTrue(np.all(np.isfinite(prediction)))

    def test_update_weights(self):
        """Test model training with sample data."""
        target_q_values = np.array([[1.0, 0.0, 0.0]])
        history = self.model.update_weights(self.valid_state, target_q_values)
        
        self.assertIsNotNone(history)
        self.assertTrue('loss' in history.history)
        self.assertTrue(len(history.history['loss']) > 0)

    def test_save_load_model(self):
        """Test model persistence operations."""
        test_filename = "test_model"
        
        # Get initial predictions
        initial_prediction = self.model.predict(self.valid_state)
        
        # Save model
        model_path, weights_path = self.model.save_model(test_filename)
        
        # Verify files exist
        self.assertTrue(Path(model_path).exists())
        self.assertTrue(Path(weights_path).exists())
        
        # Create new model instance
        new_model = QNetworkModel(action_size=self.action_size)
        
        # Load saved model
        new_model.load_model(test_filename)
        
        # Compare predictions
        loaded_prediction = new_model.predict(self.valid_state)
        np.testing.assert_array_almost_equal(
            initial_prediction,
            loaded_prediction,
            decimal=5,
            err_msg="Loaded model predictions don't match original"
        )