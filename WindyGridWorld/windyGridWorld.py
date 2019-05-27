import numpy as np


class State:
    def __init__(self, state=(3, 0), rows=7, cols=10):
        self.END_STATE = (3, 7)
        self.WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.ROWS = 7
        self.COLS = 10

        self.state = state  # starting point
        self.isEnd = True if self.state == self.END_STATE else False

    def giveReward(self):
        if self.state == self.END_STATE:
            return 1
        else:
            return 0

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        ------------------
        0  | 1 | 2| 3| ...
        1  |
        2  |
        ...|
        return next position on board based on wind strength of that column
        (according to the book, the number of steps shifted upward is based on the current state)
        """
        currentWindy = self.WIND[self.state[1]]

        if action == "up":
            nxtState = (self.state[0] - 1 - currentWindy, self.state[1])
        elif action == "down":
            nxtState = (self.state[0] + 1 - currentWindy, self.state[1])
        elif action == "left":
            nxtState = (self.state[0] - currentWindy, self.state[1] - 1)
        else:
            nxtState = (self.state[0] - currentWindy, self.state[1] + 1)

        # if next state is legal
        positionRow, positionCol = 0, 0
        if (nxtState[0] >= 0) and (nxtState[0] <= (self.ROWS - 1)):
            positionRow = nxtState[0]
        else:
            positonRow = self.state[0]

        if (nxtState[1] >= 0) and (nxtState[1] <= (self.COLS - 1)):
            positionCol = nxtState[1]
        else:
            positionCol = self.state[1]
        # if bash into walls
        return (positionRow, positionCol)

    def showBoard(self):
        self.board = np.zeros([self.ROWS, self.COLS])
        self.board[self.state] = 1
        self.board[self.END_STATE] = -1

        for i in range(self.ROWS):
            print('-----------------------------------------')
            out = '| '
            for j in range(self.COLS):
                if self.board[i, j] == 1:
                    token = 'S'
                if self.board[i, j] == -1:
                    token = 'G'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------------------------------')


class Agent:

    def __init__(self, lr=0.2, exp_rate=0.3):
        self.END_STATE = (3, 7)
        self.START_STATE = (3, 0)
        self.ROWS = 7
        self.COLS = 10

        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = lr
        self.exp_rate = exp_rate

        # initial Q values
        self.Q_values = {}
        for i in range(self.ROWS):
            for j in range(self.COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        # update State
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                if i % 5 == 0:
                    print("round", i)
                # back propagate
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append([(self.State.state), action])
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)


if __name__ == "__main__":
    print("training ...")
    ag = Agent(exp_rate=0.3)
    ag.play(50)

    print("playing ...")
    ag_op = Agent(exp_rate=0)
    ag_op.Q_values = ag.Q_values

    while not ag_op.State.isEnd:
        action = ag_op.chooseAction()
        print("current state {}, action {}".format(ag_op.State.state, action))
        ag_op.State = ag_op.takeAction(action)