import pypokergui.engine_wrapper as Engine
from models.q_network_model import QNetworkModel
from players.q_network_model_player_setup import QLearningPokerPlayer
import numpy as np

MAX_GAMES = 10

def train_poker_bots():
    # Initialize two bots
    bot1 = QLearningPokerPlayer('bot1')
    bot2 = QLearningPokerPlayer('bot2')
        # Initialize game objects
    bot1.uuid = 'bot1'
    bot2.uuid = 'bot2'
    bot1.receive_game_start_message({'uuid': 'bot1'})
    bot2.receive_game_start_message({'uuid': 'bot2'})
    # Game config
    config = Engine.gen_game_config(
        max_round=100,
        initial_stack=200,
        small_blind=1,
        ante=0,
        blind_structure=None
    )
    
    for game_num in range(MAX_GAMES):
        print(f"\nStarting Game {game_num + 1}")
        
        # Set up new game
        engine = Engine.EngineWrapper()
        players_info = Engine.gen_players_info(
            ['bot1', 'bot2'],
            ['Bot 1', 'Bot 2']
        )
        
        messages = engine.start_game(players_info, config)
        
        # Game loop
        while True:
            ask_uuid, ask_message = messages[-1]
            
            # Check if game ended
            if ask_message['message']['message_type'] == 'game_result_message':
                result = ask_message['message']
                print(f"Game {game_num + 1} finished")
                print(f"Winners: {result}")
                break
                
            # Get current player
            current_bot = bot1 if ask_uuid == 'bot1' else bot2
            
            # Get action from bot
            action, amount = current_bot.declare_action(
                ask_message['message']['valid_actions'],
                ask_message['message']['hole_card'],
                ask_message['message']['round_state']
            )
            
            # Update game state
            messages = engine.update_game(action, amount)
    bot1.model.save_model('bot1')
    bot2.model.save_model('bot2')
    print("\nTraining complete!")

if __name__ == "__main__":
    train_poker_bots()