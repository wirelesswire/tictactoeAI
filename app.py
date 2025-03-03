from flask import Flask, render_template_string, jsonify
from game import TicTacToe
from ai_model import DQLAgent, train_ai
# import requests
from flask import request
from config import MODEL_PATH
import os

app = Flask(__name__)
game = None
trained_agent = None

# Load trained model if it exists
if os.path.exists(MODEL_PATH):
    trained_agent = DQLAgent()
    trained_agent.load_model(MODEL_PATH)

BOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Tic Tac Toe</title>
    <style>
        .board { display: grid; grid-template-columns: repeat(3, 100px); gap: 5px; margin: 20px auto; }
        .cell { width: 100px; height: 100px; border: 2px solid #333; display: flex; align-items: center; justify-content: center; font-size: 40px; cursor: pointer; }
        .cell:hover { background-color: #f0f0f0; }
        .container { text-align: center; max-width: 600px; margin: 0 auto; padding: 20px; }
        .status { margin: 20px 0; font-size: 24px; }
        .controls { margin: 20px 0; }
        button { padding: 10px 20px; margin: 0 10px; font-size: 16px; cursor: pointer; }
        .training-controls { margin: 20px 0; padding: 20px; border: 1px solid #ccc; border-radius: 5px; }
        .training-controls input { margin: 0 10px; padding: 5px; }
        .opponent-indicator { background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; font-weight: bold; }
        .opponent-minimax { color: #e74c3c; }
        .opponent-dql { color: #2ecc71; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Tic Tac Toe</h1>
        {% if current_mode %}
        <div class="opponent-indicator {% if current_mode == 'dql' %}opponent-dql{% else %}opponent-minimax{% endif %}">
            Current Opponent: {{ 'Trained AI (DQL)' if current_mode == 'dql' else 'Minimax AI (Unbeatable)' }}
        </div>
        {% endif %}
        <div class="controls">
            <button onclick="startGame('minimax')">Play vs Minimax AI</button>
            <button onclick="startGame('dql')" id="dqlButton" {% if not trained_agent %}disabled{% endif %}>Play vs Trained AI</button>
        </div>
        <div class="training-controls">
            <h3>Train AI Model</h3>
            <input type="number" id="episodes" value="1000" min="100" step="100">
            <button onclick="trainAI()">Start Training</button>
            <div id="trainingStatus"></div>
        </div>
        <div class="status">{{ status }}</div>
        <div class="board">
            {% for i in range(3) %}
                {% for j in range(3) %}
                    <div class="cell" onclick="{% if not game_over %}makeMove({{ i }}, {{ j }}){% endif %}">
                        {{ board[i][j] }}
                    </div>
                {% endfor %}
            {% endfor %}
        </div>
    </div>
    <script>
        function startGame(mode) {
            fetch('/start-game/' + mode, {
                method: 'POST'
            })
            .then(response => response.text())
            .then(html => {
                document.body.innerHTML = html;
                // Always enable the DQL button if the model is trained
                const dqlButton = document.getElementById('dqlButton');
                if (dqlButton) {
                    dqlButton.disabled = false;
                }
            });
        }

        function makeMove(row, col) {
            const currentMode = document.querySelector('.opponent-indicator').classList.contains('opponent-dql') ? 'dql' : 'minimax';
            fetch(`/make-move/${row}/${col}?mode=${currentMode}`, {
                method: 'POST'
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error);
                    });
                }
                return response.text();
            })
            .then(html => {
                document.body.innerHTML = html;
                // Keep DQL button enabled after moves
                const dqlButton = document.getElementById('dqlButton');
                if (dqlButton) {
                    dqlButton.disabled = false;
                }
            })
            .catch(error => {
                const statusDiv = document.querySelector('.status');
                if (statusDiv) {
                    statusDiv.textContent = error.message;
                }
            });
        }

        function trainAI() {
            const episodes = document.getElementById('episodes').value;
            const statusDiv = document.getElementById('trainingStatus');
            const dqlButton = document.getElementById('dqlButton');
            
            statusDiv.textContent = 'Training in progress...';
            fetch('/train-ai', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({episodes: parseInt(episodes)})
            })
            .then(response => response.json())
            .then(data => {
                statusDiv.textContent = data.message;
                dqlButton.disabled = false;
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(
        BOARD_TEMPLATE,
        board=[['' for _ in range(3)] for _ in range(3)],
        status="Welcome! Choose a game mode to start",
        game_over=False
    )

@app.route('/start-game/<mode>', methods=['POST'])
def start_game(mode):
    global game
    game = TicTacToe()
    return render_template_string(
        BOARD_TEMPLATE,
        board=game.get_board_state(),
        status=f"Game Started - X's turn vs {'Trained AI' if mode == 'dql' else 'Minimax AI'}",
        game_over=False,
        current_mode=mode
    )

@app.route('/make-move/<int:row>/<int:col>', methods=['POST'])
def make_move(row, col):
    if not game:
        return jsonify({'error': 'Game not started'}), 400

    # Player move
    if not game.make_move(row, col):
        return jsonify({'error': 'Invalid move'}), 400

    # Check game status after player move
    winner = game.check_winner()
    if winner:
        status = "Game Over - It's a Draw!" if winner == 'Draw' else f"Game Over - {winner} Wins!"
        return render_template_string(
            BOARD_TEMPLATE,
            board=game.get_board_state(),
            status=status,
            game_over=True,
            current_mode=request.args.get('mode', 'minimax')
        )

    # AI move
    current_mode = request.args.get('mode', 'minimax')
    if current_mode == 'dql' and trained_agent and game.current_player == 'O':
        state = trained_agent.get_state(game.get_board_state())
        ai_row, ai_col = trained_agent.act(state, game.get_valid_moves())
    else:
        ai_row, ai_col = game.get_ai_move()
    
    game.make_move(ai_row, ai_col)

    # Check game status after AI move
    winner = game.check_winner()
    game_over = winner is not None

    opponent_type = 'Trained AI' if current_mode == 'dql' else 'Minimax AI'
    status = f"Game in progress - {game.current_player}'s turn vs {opponent_type}"
    if winner == 'Draw':
        status = "Game Over - It's a Draw!"
    elif winner:
        status = f"Game Over - {winner} Wins!"

    return render_template_string(
        BOARD_TEMPLATE,
        board=game.get_board_state(),
        status=status,
        game_over=game_over,
        current_mode=current_mode
    )

@app.route('/train-ai', methods=['POST'])
def train_ai_endpoint():
    global trained_agent
    episodes = request.json.get('episodes', 1000)
    trained_agent = train_ai(episodes)
    # Save the trained model
    trained_agent.save_model(MODEL_PATH)
    return jsonify({'message': f'Training completed after {episodes} episodes!'})

if __name__ == '__main__':
    app.run(debug=True)