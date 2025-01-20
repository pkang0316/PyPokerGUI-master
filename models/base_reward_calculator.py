class BaseRewardCalculator:
    """
    A general-purpose reward calculator for reinforcement learning in poker.
    This base class defines the interface and common reward calculations
    that would be useful across different types of models.
    """
    def __init__(self, config=None):
        # Default configuration parameters
        self.default_config = {
            'position_multiplier': 1.2,      # Bonus for being in position
            'pot_equity_scale': 0.5,         # Scale factor for pot equity rewards
            'stack_preservation_bonus': 0.2,  # Bonus for preserving stack
            'value_bet_river_bonus': 0.3,    # Bonus for river value bets
            'value_bet_turn_bonus': 0.2,     # Bonus for turn value bets
            'small_loss_threshold': 0.1,     # Threshold for small loss bonus
            'medium_loss_threshold': 0.25,   # Threshold for medium loss bonus
            'pot_commitment_threshold': 3,    # SPR threshold for pot commitment
        }
        # Override defaults with provided config
        self.config = {**self.default_config, **(config or {})}

    def calculate_reward(self, current_state, action, next_state, is_terminal=False):
        """
        Main entry point for reward calculation. This orchestrates the overall
        reward calculation process.
        
        Parameters:
            current_state: Dict containing the current game state
            action: Dict containing the action taken
            next_state: Dict containing the resulting game state
            is_terminal: Boolean indicating if this is the end of an episode
        """
        if is_terminal:
            return self.calculate_terminal_reward(current_state, action, next_state)
        return self.calculate_intermediate_reward(current_state, action, next_state)

    def calculate_terminal_reward(self, current_state, action, next_state):
        """
        Calculate reward for terminal states based on fundamental poker concepts.
        """
        reward = 0
        try:
            # Calculate chip differential
            initial_stack = self._get_stack_size(current_state)
            final_stack = self._get_stack_size(next_state)
            chip_differential = final_stack - initial_stack
            
            # Base reward normalized by typical stack size
            base_reward = chip_differential / 1000.0
            
            # Position multiplier
            if self._is_in_position(current_state):
                base_reward *= self.config['position_multiplier']
            
            # Stack preservation bonus
            if chip_differential < 0:
                loss_percentage = abs(chip_differential) / initial_stack
                if loss_percentage < self.config['small_loss_threshold']:
                    base_reward += self.config['stack_preservation_bonus']
                elif loss_percentage < self.config['medium_loss_threshold']:
                    base_reward += self.config['stack_preservation_bonus'] / 2
            
            # Value betting bonuses
            if self._made_value_bet_on_river(current_state):
                base_reward += self.config['value_bet_river_bonus']
            if self._made_value_bet_on_turn(current_state):
                base_reward += self.config['value_bet_turn_bonus']
            
            reward = base_reward
            
        except Exception as e:
            print(f"Error in calculate_terminal_reward: {e}")
        
        return reward

    def calculate_intermediate_reward(self, current_state, action, next_state):
        """
        Calculate reward for intermediate states based on poker strategy concepts.
        """
        reward = 0
        try:
            # Pot equity component
            pot_size = self._get_pot_size(current_state)
            hand_strength = self._estimate_hand_strength(current_state)
            pot_equity = pot_size * hand_strength
            reward += (pot_equity / 1000) * self.config['pot_equity_scale']
            
            # Position consideration
            if self._is_in_position(current_state):
                reward *= self.config['position_multiplier']
            
            # Stack-to-pot ratio consideration
            spr = self._get_stack_to_pot_ratio(current_state)
            if spr < self.config['pot_commitment_threshold']:
                reward *= hand_strength * 1.5
                
        except Exception as e:
            print(f"Error in calculate_intermediate_reward: {e}")
            
        return reward

    # Protected helper methods for state information extraction
    def _get_stack_size(self, state):
        """Extract stack size from state."""
        raise NotImplementedError("Subclasses must implement _get_stack_size")

    def _get_pot_size(self, state):
        """Extract pot size from state."""
        raise NotImplementedError("Subclasses must implement _get_pot_size")

    def _is_in_position(self, state):
        """Determine if player is in position."""
        raise NotImplementedError("Subclasses must implement _is_in_position")

    def _estimate_hand_strength(self, state):
        """Estimate current hand strength."""
        raise NotImplementedError("Subclasses must implement _estimate_hand_strength")

    def _made_value_bet_on_river(self, state):
        """Check if made value bet on river."""
        raise NotImplementedError("Subclasses must implement _made_value_bet_on_river")

    def _made_value_bet_on_turn(self, state):
        """Check if made value bet on turn."""
        raise NotImplementedError("Subclasses must implement _made_value_bet_on_turn")

    def _get_stack_to_pot_ratio(self, state):
        """Calculate stack to pot ratio."""
        try:
            pot_size = self._get_pot_size(state)
            if pot_size == 0:
                return float('inf')
            return self._get_stack_size(state) / pot_size
        except Exception as e:
            print(f"Error in _get_stack_to_pot_ratio: {e}")
            return float('inf')