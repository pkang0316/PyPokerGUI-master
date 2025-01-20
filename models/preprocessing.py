# preprocessing.py

def preprocess_card(card):
    rank_dict = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_dict = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    rank_onehot = [0] * 13
    suit_onehot = [0] * 4
    rank_onehot[rank_dict[card[1]]] = 1
    suit_onehot[suit_dict[card[0]]] = 1
    return rank_onehot + suit_onehot

def preprocess_action(action):
    action_dict = {'fold': 0, 'call': 1, 'raise': 2}
    return action_dict.get(action, -1)

def preprocess_betting_round(round_state):
    return round_state['action_histories']

def preprocess_stack_sizes(round_state, num_players):
    stack_sizes = [0] * num_players
    for i, player_info in enumerate(round_state['seats']):
        stack_sizes[i] = player_info['stack'] / (player_info['stack'] + round_state['pot']['main']['amount'])
    return stack_sizes

def preprocess_game_stage(round_state):
    stage_dict = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    return stage_dict[round_state['street']]