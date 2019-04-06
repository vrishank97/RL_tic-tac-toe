import numpy as np

class Board:
    def __init__(self):
        self.board = []
        self.done = False
        self.reset()

    def create_board(self):
        return (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))

    def reset(self):
        self.board = self.create_board()
        self.done = False
        return self.board

    def show_valid(self):
        return np.ravel(self.board) == 0

    def row_win(self, player):
        board = self.board
        for x in range(len(board)): 
            win = True
              
            for y in range(len(board)): 
                if board[x, y] != player: 
                    win = False
                    continue
                      
            if win == True: 
                return(win) 
        return(win)

    def col_win(self, player):
        board = self.board
        for x in range(len(board)): 
            win = True
              
            for y in range(len(board)): 
                if board[y][x] != player: 
                    win = False
                    continue
                      
            if win == True: 
                return(win) 
        return(win)

    def diag_win(self, player):
        board = self.board
        win = True
          
        for x in range(len(board)): 
            if board[x, x] != player: 
                win = False

        if board[0, 2] ==player:
            if board[1, 1] ==player:
                if board[2, 0] ==player:
                    win = True
        return(win)

    def evaluate(self):
        board = self.board
        winner = 0
          
        for player in [-1, 1]: 
            if (self.row_win(player) or self.col_win(player) or self.diag_win(player)):
                winner = player 
                  
        if np.all(board != 0) and winner == 0: 
            winner = 0
        return winner

    def step1(self, action):
        flag2 = 0
        if not self.done:
            if(self.show_valid()[action]): 
                self.board[action//3][action%3]=-1
            else:
                # print("Invalid Move by player 1 - {}".format(action))
                # print(self.board)
                for i in range(9):
                    if self.show_valid()[i]:
                        flag2=1
                        # print("Valid move was {}".format(i))
                        self.board[i//3][i%3]=-1
                    break


        state = self.evaluate()
        if state is 0:
            self.done = False
        if state is -1:
            self.done = True
        if state is -1:
            self.done = True
        info = None

        if not self.done:
            flag = 0
            for i in range(9):
                if self.show_valid()[i]:
                    flag = 1
            if flag == 0:
                self.done = True
        return self.board, self.evaluate(), self.done, info

    def step2(self, action):
        flag2 = 0
        if not self.done:
            if(self.show_valid()[action]):
                self.board[action//3][action%3]=1
            else:
                # print("Invalid Move by player 2 - {}".format(action))
                # print(self.board)
                for i in range(9):
                    if self.show_valid()[i]:
                        flag2=1
                        # print("Valid move was {}".format(i))
                        self.board[i//3][i%3]=1
                    break


        state = self.evaluate()
        if state is 0:
            self.done = False
        if state is 1:
            self.done = True
        if state is -1:
            self.done = True
        info = None

        if not self.done:
            flag = 0
            for i in range(9):
                if self.show_valid()[i]:
                    flag = 1
            if flag == 0:
                self.done = True
        return self.board, self.evaluate(), self.done, info

