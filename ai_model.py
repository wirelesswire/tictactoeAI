import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from game import TicTacToe

class TicTacNet(nn.Module):
    def __init__(self):
        super(TicTacNet, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # 3x3 board flattened
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 9)   # 9 possible moves

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQLAgent:
    def __init__(self, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.model = TicTacNet()
        self.target_model = TicTacNet()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

    def get_state(self, board):
        # Convert board to numerical values: '' -> 0, 'X' -> 1, 'O' -> -1
        state = [[0 if cell == '' else 1 if cell == 'X' else -1 for cell in row] for row in board]
        return torch.FloatTensor(state).flatten()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        with torch.no_grad():
            q_values = self.model(state)
            # Mask invalid moves with large negative values
            valid_moves_mask = torch.ones(9) * float('-inf')
            for move in valid_moves:
                valid_moves_mask[move[0] * 3 + move[1]] = 0
            q_values += valid_moves_mask
            action_idx = q_values.argmax().item()
            return (action_idx // 3, action_idx % 3)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([s[0] for s in batch])
        actions = torch.tensor([(s[1][0] * 3 + s[1][1]) for s in batch])
        rewards = torch.tensor([s[2] for s in batch])
        next_states = torch.stack([s[3] for s in batch])
        dones = torch.tensor([s[4] for s in batch], dtype=torch.float)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())

def train_ai(episodes, opponent_type='self'):
    agent = DQLAgent()
    game = TicTacToe()
    wins = 0
    
    for episode in range(episodes):
        game = TicTacToe()
        state = agent.get_state(game.get_board_state())
        done = False
        
        while not done:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
                
            action = agent.act(state, valid_moves)
            game.make_move(action[0], action[1])
            
            # Opponent's move
            if opponent_type == 'minimax':
                if game.check_winner() is None and game.get_valid_moves():
                    ai_row, ai_col = game.get_ai_move()
                    game.make_move(ai_row, ai_col)
            elif opponent_type == 'self':
                if game.check_winner() is None and game.get_valid_moves():
                    opponent_action = agent.act(agent.get_state(game.get_board_state()), game.get_valid_moves())
                    game.make_move(opponent_action[0], opponent_action[1])
            
            next_state = agent.get_state(game.get_board_state())
            winner = game.check_winner()
            
            # Reward calculation
            if winner == 'X':
                reward = 1
                wins += 1
                done = True
            elif winner == 'O':
                reward = -1
                done = True
            elif winner == 'Draw':
                reward = 0.5
                done = True
            else:
                reward = 0
                done = False
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            
        if episode % 100 == 0:
            agent.update_target_model()
            win_rate = wins / (episode + 1) * 100
            print(f'Episode: {episode}, Win Rate: {win_rate:.2f}%, Epsilon: {agent.epsilon:.2f}')
    
    return agent