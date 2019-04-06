#--------------------------------------------------------------------
# Tic Tac Toe game
#
# Designed so that the computer always wins or ties
# Uses the minimax algorithm with alpha beta pruning to calculate the
#  next best move
#
# Written in Flask framework and AngularJS for the frontend
#--------------------------------------------------------------------
from flask import Flask, render_template, jsonify, request
from game import Game

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


from board import Board
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, num_classes)  
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DQN(9, 9).to(device)
#model = torch.load("player2.pth")
model.load_state_dict(torch.load("player2dict.pth", map_location=device))
state = [[-1, 0, 0], [0, 0, 0], [0, 0, 0]]
state = torch.tensor([state], device=device, dtype=torch.float32).view(-1)
with torch.no_grad():
    action = model(state)
print(action.numpy().argmax())

@app.route('/move', methods=['POST'])
def move():
    post = request.get_json()

    game = Game()
    game.board = post.get('board')
    game.player = post.get('player')
    game.computer = post.get('computer')

    # Check if player won
    if game.has_won(game.player):
        return jsonify(tied = False, computer_wins = False, player_wins = True, board = game.board)
    elif game.is_board_full():
        return jsonify(tied = True, computer_wins = False, player_wins = False, board = game.board)

    # Calculate computer move
    computer_move = game.calculate_move()
    temp  = game.board
    for i in range(3):
        for j in range(3):
            if temp[i][j]=='X':
                temp[i][j]=-1
            if temp[i][j]=='O':
                temp[i][j]=1
            if temp[i][j]==' ':
                temp[i][j]=0
    print(temp)
    temp2 = Board()
    temp2.board = temp

    state = torch.tensor([temp], device=device, dtype=torch.float32).view(-1)
    print(temp2.show_valid())
    valid_actions = torch.tensor([1*temp2.show_valid()], device=device, dtype=torch.float32)
    with torch.no_grad():
        action = model(state)*valid_actions
    action = action.numpy().argmax()

    #game.agent.select_action(temp2.board, temp2.show_valid())
    #print()

    # Make the next move
    game.make_computer_move(action//3, action%3)
    print(computer_move['row'])
    print(computer_move['col'])
    print(action//3, int(action%3))

    # Check if computer won
    if game.has_won(game.computer):
        return jsonify(computer_row = int(action//3), computer_col = int(action%3),
                       computer_wins = True, player_wins = False, tied=False, board = game.board)
    # Check if game is over
    elif game.is_board_full():
        return jsonify(computer_row = int(action//3), computer_col = int(action%3),
                       computer_wins = False, player_wins = False, tied=True, board=game.board)

    # Game still going
    return jsonify(computer_row = int(action//3), computer_col = int(action%3),
                   computer_wins = False, player_wins = False, board = game.board)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', debug=True)