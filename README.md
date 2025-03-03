# TicTacs - A Modern Take on Tic-Tac-Toe

TicTacs is an  web-based implementation of Tic-Tac-Toe with a  twist - each player can only have three pieces on the board at a time! When a fourth piece is placed, the oldest piece is removed, creating a dynamic and strategic gameplay experience.

## Features

- Classic 3x3 game board
- Two AI opponents to choose from:
  - Minimax AI: A perfect player using the minimax algorithm with alpha-beta pruning
  - Deep Q-Learning AI: A trained neural network that learns from experience
- Unique "three pieces only" rule that adds an extra layer of strategy
- Clean and intuitive web interface
- Real-time game status updates

## Game Rules

1. Players take turns placing their pieces (X or O) on the board
2. Each player can only have three pieces on the board at any time
3. When a player places a fourth piece, their oldest piece is removed
4. Win by getting three of your pieces in a row (horizontally, vertically, or diagonally)
5. The game ends in a draw if no player achieves three in a row

## Technical Stack

- Backend: Python with Flask web framework
- AI Implementation:
  - Minimax algorithm with alpha-beta pruning for perfect play
  - PyTorch for the Deep Q-Learning neural network
- Frontend: HTML, JavaScript, and CSS

## Setup Instructions

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:
   ```bash
   pip install flask torch numpy
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open your web browser and navigate to `http://localhost:5000`

## Playing Against AI

- Choose your preferred AI opponent at the start of the game
- Minimax AI provides perfect gameplay
- Deep Q-Learning AI offers a more varied playing style based on its training

## Project Structure

- `app.py`: Main Flask application and route handlers
- `game.py`: Core game logic and Minimax AI implementation
- `ai_model.py`: Deep Q-Learning neural network implementation
- `config.py`: Configuration settings
- `trained_model.pth`: Pre-trained neural network weights