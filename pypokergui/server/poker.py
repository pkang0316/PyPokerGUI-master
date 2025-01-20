import os
import sys
import yaml
import uuid
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
from tornado.options import define, options

import pypokerengine.utils.action_utils as AU
import pypokergui.server.game_manager as GM
import pypokergui.server.message_manager as MM

# Define command-line options
define("port", default=8888, help="run on the given port", type=int)
define("config", default=None, help="path to game config", type=str)
define("speed", default="moderate", help="how fast game progress", type=str)

# Application class that sets up routing for handling requests
class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", PokerRequestHandler),
            (r"/pokersocket", PokerWebSocketHandler),
        ]
        settings = dict(
            cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            static_path=os.path.join(os.path.dirname(__file__), "static"),  # Serving static files
            template_path=os.path.join(os.path.dirname(__file__), "templates"),  # Template files
            xsrf_cookies=True,
        )
        super(Application, self).__init__(handlers, debug=True, **settings)

# Handler for the main game page
class PokerRequestHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", config=global_game_manager, registered=False)

# WebSocket handler for player interactions
class PokerWebSocketHandler(tornado.websocket.WebSocketHandler):
    sockets = set()

    def get_compression_options(self):
        # Enables compression for WebSocket communication
        return {}

    def open(self):
        self.uuid = str(uuid.uuid4())  # Unique identifier for each player
        PokerWebSocketHandler.sockets.add(self)

    def on_close(self):
        PokerWebSocketHandler.sockets.remove(self)
        if global_game_manager.get_human_player_info(self.uuid):
            global_game_manager.remove_human_player_info(self.uuid)
            MM.broadcast_config_update(self, global_game_manager, self.sockets)

    def on_message(self, message):
        # Handle messages received from players
        js = tornado.escape.json_decode(message)
        message_type = js['type']
        
        if 'action_new_member' == message_type:
            global_game_manager.join_human_player(js['name'], self.uuid)
            MM.broadcast_config_update(self, global_game_manager, self.sockets)
        elif 'action_start_game' == message_type:
            if global_game_manager.is_playing_poker:
                MM.alert_server_restart(self, self.uuid, self.sockets)
            else:
                global_game_manager.start_game()
                MM.broadcast_start_game(self, global_game_manager, self.sockets)
                MM.broadcast_update_game(self, global_game_manager, self.sockets, MODE_SPEED)
                if self._is_next_player_ai(global_game_manager):
                    self._progress_the_game_till_human()
        elif 'action_declare_action' == message_type:
            if self.uuid == global_game_manager.next_player_uuid:
                action, amount = self._correct_action(js)
                global_game_manager.update_game(action, amount)
                MM.broadcast_update_game(self, global_game_manager, self.sockets, MODE_SPEED)
                if self._is_next_player_ai(global_game_manager):
                    self._progress_the_game_till_human()
        else:
            raise Exception(f"Unexpected message [ {message} ] received")

    def _correct_action(self, data):
        try:
            data["amount"] = int(data["amount"])
        except:
            data["amount"] = -1
        players = global_game_manager.engine.current_state["table"].seats.players
        next_player_pos = global_game_manager.engine.current_state["next_player"]
        sb_amount = global_game_manager.engine.current_state["small_blind_amount"]
        actions = AU.generate_legal_actions(players, next_player_pos, sb_amount)

        if data["action"] == "fold":
            data["amount"] = 0
        elif data["action"] == "call":
            data["amount"] = actions[1]["amount"]
        else:
            legal = actions[2]["amount"]
            if legal["min"] <= data["amount"] <= legal["max"]:
                data["amount"] = data["amount"]
            else:
                data["action"] = "fold"
                data["amount"] = 0
        return data["action"], data["amount"]

    def _progress_the_game_till_human(self):
        # Progress the game until it's the human player's turn
        while self._is_next_player_ai(global_game_manager):
            if GM.has_game_finished(global_game_manager.latest_messages): break
            action, amount = global_game_manager.ask_action_to_ai_player(global_game_manager.next_player_uuid)
            global_game_manager.update_game(action, amount)
            MM.broadcast_update_game(self, global_game_manager, self.sockets, MODE_SPEED)

    def _is_next_player_ai(self, game_manager):
        uuid = game_manager.next_player_uuid
        return uuid and len(uuid) <= 2

# Game mode speed (could be set from the configuration)
MODE_SPEED = "moderate"
global_game_manager = GM.GameManager()

def setup_config(config):
    global_game_manager.define_rule(
        config['max_round'], config['initial_stack'], config['small_blind'],
        config['ante'], config['blind_structure']
    )
    for player in config['ai_players']:
        global_game_manager.join_ai_player(player['name'], player['path'])

def start_server(config_path, port, speed):
    global MODE_SPEED
    with open(config_path, "rb") as f:
        config = yaml.safe_load(f)
    setup_config(config)
    MODE_SPEED = speed
    app = Application()
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()

def main():
    tornado.options.parse_command_line()
    start_server(options.config, options.port, options.speed)

if __name__ == '__main__':
    main()
