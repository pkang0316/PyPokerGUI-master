import pypokergui.engine_wrapper as Engine
import pypokergui.ai_generator as AG

class GameManager(object):
    def __init__(self):
        self.rule = None
        self.members_info = []
        self.engine = None
        self.ai_players = {}
        self.is_playing_poker = False
        self.latest_messages = []
        self.next_player_uuid = None
        self.round_count = 0
        self.game_ended = False  # Track whether the game has ended

    def define_rule(self, max_round, initial_stack, small_blind, ante, blind_structure):
        self.rule = Engine.gen_game_config(max_round, initial_stack, small_blind, ante, blind_structure)

    def join_ai_player(self, name, setup_script_path):
        ai_uuid = str(len(self.members_info))
        self.members_info.append(gen_ai_player_info(name, ai_uuid, setup_script_path))

    def join_human_player(self, name, uuid):
        self.members_info.append(gen_human_player_info(name, uuid))

    def get_human_player_info(self, uuid):
        for info in self.members_info:
            if info["type"] == "human" and info["uuid"] == uuid:
                return info

    def remove_human_player_info(self, uuid):
        member_info = self.get_human_player_info(uuid)
        assert member_info
        self.members_info.remove(member_info)

    def start_game(self):
        assert self.rule and len(self.members_info) >= 2 and not self.is_playing_poker
        uuid_list = [member["uuid"] for member in self.members_info]
        name_list = [member["name"] for member in self.members_info]
        players_info = Engine.gen_players_info(uuid_list, name_list)
        self.ai_players = build_ai_players(self.members_info)
        self.engine = Engine.EngineWrapper()
        self.latest_messages = self.engine.start_game(players_info, self.rule)
        self.is_playing_poker = True
        self.next_player_uuid = fetch_next_player_uuid(self.latest_messages)

    def update_game(self, action, amount):
        assert len(self.latest_messages) != 0  # check that start_game has already been called
        self.latest_messages = self.engine.update_game(action, amount)
        self.next_player_uuid = fetch_next_player_uuid(self.latest_messages)

    def ask_action_to_ai_player(self, uuid):
        assert uuid in self.ai_players
        ai_player = self.ai_players[uuid]
        ask_uuid, ask_message = self.latest_messages[-1]
        assert ask_message['type'] == 'ask' and uuid == ask_uuid
        return ai_player.declare_action(
                ask_message['message']['valid_actions'],
                ask_message['message']['hole_card'],
                ask_message['message']['round_state']
        )

    def end_game(self):
        """
        Called when the game ends to finalize any last steps.
        Save weights and reinitialize for the next round.
        """
        print("Game has ended. Saving model and reinitializing for the next round.")

        # Save models for AI players before the game ends
        self.save_model_for_ai_players()

        # Reinitialize the game for the next round
        self.reinitialize_game()

        # Start a new game automatically
        self.start_game()

    def save_model_for_ai_players(self):
        """
        Save the models for each AI player.
        """
        for ai_uuid, ai_player in self.ai_players.items():
            try:
                ai_player.model.save_model()  # Assuming `save_model()` method exists for AI players
            except:
                print(f"Failed to save model for player [ {ai_uuid} ]")

    def reinitialize_game(self):
        """
        Reinitialize the game by clearing current state and loading saved weights.
        """
        # Clear current state and reinitialize game objects
        self.is_playing_poker = False
        self.latest_messages = []
        self.next_player_uuid = None
        self.round_count = 0

        # Reload AI player models
        self.ai_players = {}
        for member in self.members_info:
            if member["type"] == "human": continue
            ai_uuid = member["uuid"]
            ai_player = self.reload_ai_player(member["setup_script_path"])
            self.ai_players[ai_uuid] = ai_player

    def reload_ai_player(self, setup_script_path):
        """
        Reloads an AI player with the saved weights.
        """
        if not AG.healthcheck(setup_script_path, quiet=True):
            raise Exception(f"Failed to setup AI from [ {setup_script_path} ]")
        setup_method = AG._import_setup_method(setup_script_path)
        ai_player = setup_method()
        ai_player.load_model()  # Load saved model weights
        return ai_player


def fetch_next_player_uuid(new_messages):
    if not has_game_finished(new_messages):
        ask_uuid, ask_message = new_messages[-1]
        assert ask_message['type'] == 'ask'
        return ask_uuid

def has_game_finished(new_messages):
    _uuid, last_message = new_messages[-1]
    return "game_result_message" == last_message['message']['message_type']

def build_ai_players(members_info):
    holder = {}
    for member in members_info:
        if member["type"] == "human": continue
        holder[member["uuid"]] = _build_ai_player(member["setup_script_path"])
    return holder

def _build_ai_player(setup_script_path):
    if not AG.healthcheck(setup_script_path, quiet=True):
        raise Exception(f"Failed to setup AI from [ {setup_script_path} ]")
    setup_method = AG._import_setup_method(setup_script_path)
    return setup_method()

def gen_ai_player_info(name, uuid, setup_script_path):
    info = _gen_base_player_info("ai", name, uuid)
    info["setup_script_path"] = setup_script_path
    return info

def gen_human_player_info(name, uuid):
    return _gen_base_player_info("human", name, uuid)

def _gen_base_player_info(player_type, name, uuid):
    return {
            "type": player_type,
            "name": name,
            "uuid": uuid
            }
