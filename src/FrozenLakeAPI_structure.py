from flask import Flask, request, json
import gymnasium as gym
from uuid import uuid4
from typing import Dict, Any

class FrozenLakeAPI:
    def __init__(self) -> None:
        self.app: Flask = Flask(__name__)
        self.app.debug = True
        self.games: Dict[str, gym.Env] = {}

    def run_server(self) -> None:
        self.app.route('/new_game', methods=['GET'])(self.new_game)
        self.app.route('/step', methods=['POST'])(self.step)
        self.app.route('/reset', methods=['POST'])(self.reset)
        self.app.run(host="localhost", port=5005, threaded=True)

    def new_game(self) -> Any:
        """
        Creates a new FrozenLake environment and assigns a unique game uuid to it.
        Returns:
            Any: A JSON response with the game_id and the status code or an error.
        """
        game_id = str(uuid4())
        env = gym.make("FrozenLake-v1", map_name='4x4', is_slippery=True)
        self.games[game_id] = env
        response_data = json.dumps({
            'game_id': game_id,
            'success': True
            })
        return response_data, 200, {'ContentType': 'application/json'}

    def reset(self) -> Any:
        """
        Resets the specified FrozenLake environment based on game_id in the request.
        Returns:
            Any: A JSON response with the updated observation or an error.
        """

        game_id = request.json['game_id']
        if game_id not in self.games:
            return json.dumps({'error': f'Game with ID {game_id} not found.'}), 404, {'ContentType': 'application/json'}
        env = self.games[game_id]
        observation, _ = env.reset()
        response_data = json.dumps({
            'observation': [observation],
            'success': True
        })
        return response_data, 200, {'ContentType': 'application/json'}

    def step(self) -> Any:
        """
        Takes a step in the specified FrozenLake environment using the given action.
        Returns:
            Any: A JSON response with observation, reward, done, and info or an error.
        """
        data = request.get_json()
        game_id = data.get('game_id')
        action = data.get('action')
        if game_id not in self.games:
            return json.dumps({
                'error': f'Game with ID {game_id} not found.'}), 404, {'ContentType': 'application/json'}
        env = self.games[game_id]
        observation, reward, done,_, info = env.step(action)
        response_data = json.dumps({
        'observation': [observation],
        'reward': reward,
        'done': done,
        'info': info,
        'success': True
        })
        return response_data , 200, {'ContentType': 'application/json'}

if __name__ == '__main__':
    emulation_api = FrozenLakeAPI()
    emulation_api.run_server()
