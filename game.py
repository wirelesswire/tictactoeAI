import random
from typing import List, Tuple, Optional

class TicTacToe:
    def __init__(self):
        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.x_moves = []
        self.o_moves = []

    def make_move(self, row: int, col: int) -> bool:
        if self.board[row][col] == '':
            # Remove oldest piece if we already have 3
            moves_list = self.x_moves if self.current_player == 'X' else self.o_moves
            if len(moves_list) >= 3:
                old_row, old_col = moves_list.pop(0)
                self.board[old_row][old_col] = ''
            
            # Add new piece
            self.board[row][col] = self.current_player
            moves_list.append((row, col))
            
            # Switch player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == '']

    def check_winner(self) -> Optional[str]:
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != '':
                return row[0]

        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != '':
                return self.board[0][col]

        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != '':
            return self.board[0][2]

        # Check for draw
        if not any('' in row for row in self.board):
            return 'Draw'

        return None

    def get_ai_move(self) -> Tuple[int, int]:
        # For minimax AI, we want to maximize if it's X's turn, minimize if it's O's turn
        _, best_move = self.minimax(0, self.current_player == 'X')
        # If no best move is found (shouldn't happen with proper minimax), choose random
        if not best_move:
            return random.choice(self.get_valid_moves())
        return best_move

    def get_board_state(self) -> List[List[str]]:
        return self.board

    def minimax(self, depth: int, is_maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[int, Optional[Tuple[int, int]]]:
        winner = self.check_winner()
        if winner == 'X':
            return 1, None
        elif winner == 'O':
            return -1, None
        elif winner == 'Draw':
            return 0, None

        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return 0, None

        best_move = None
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                row, col = move
                self.board[row][col] = 'X'
                eval_score, _ = self.minimax(depth + 1, False, alpha, beta)
                self.board[row][col] = ''
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                row, col = move
                self.board[row][col] = 'O'
                eval_score, _ = self.minimax(depth + 1, True, alpha, beta)
                self.board[row][col] = ''
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move