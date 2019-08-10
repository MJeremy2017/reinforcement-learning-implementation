import numpy as np
import matplotlib.pyplot as plt
from TileCoding import *

ACTIONS = [0, 1]


class ValueFunction:

    def __init__(self, alpha=0.01, numOfTilings=8, maxSize=2048):
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings

        # divide step size equally to each tiling
        self.alpha = alpha / numOfTilings  # learning rate for each tile

        self.hashTable = IHT(maxSize)

        # weight for each tile
        self.weights = np.zeros(maxSize)

        # position and velocity needs scaling to satisfy the tile software
        self.serverScale = self.numOfTilings / 10.0  # 10 servers
        self.priorityScale = self.numOfTilings / 3.0  # 4 kinds of priorities

    # get indices of active tiles for given state and action
    def getActiveTiles(self, n_server, priority, action):
        activeTiles = tiles(self.hashTable, self.numOfTilings,
                            [self.serverScale * n_server, self.priorityScale * priority],
                            [action])
        return activeTiles

    # estimate the value of given state and action
    def value(self, state, action):
        n_server, priority = state
        activeTiles = self.getActiveTiles(n_server, priority, action)
        return np.sum(self.weights[activeTiles])  # /self.numOfTilings

    # learn with given state, action and target
    def update(self, state, action, delta):
        n_server, priority = state
        activeTiles = self.getActiveTiles(n_server, priority, action)

        delta *= self.alpha
        for activeTile in activeTiles:
            self.weights[activeTile] += delta

    def stateValue(self, state):
        if state[0] == 0:
            # no server available
            return self.value(state, 0)
        values = [self.value(state, a) for a in ACTIONS]
        return max(values)


class ServerAcess:
    def __init__(self, exp_rate=0.3, lr=0.1, beta=0.01):
        self.n_server = 10
        self.free_prob = 0.06
        self.priorities = range(4)
        self.actions = ACTIONS  # 0: reject; 1: accept
        self.state = (0, 0)  # (num_servers, priority)

        self.exp_rate = exp_rate
        self.lr = lr
        self.beta = beta

    #         self.alpha = alpha

    def numFreeServers(self):
        n = 0
        n_free_server = self.state[0]
        n_busy_server = self.n_server - n_free_server
        for _ in range(n_busy_server):
            if np.random.uniform(0, 1) <= 0.06:
                n += 1
        n_free_server += n
        self.state = (n_free_server, self.state[1])
        return n_free_server

    def chooseAction(self, valueFunc):
        n_free_server = self.numFreeServers()
        if n_free_server == 0:
            return 0
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            values = {}
            for a in self.actions:
                v = valueFunc.value(self.state, a)
                values[a] = v
            action = np.random.choice([k for k, v in values.items() if v == max(values.values())])
        return action

    def nxtState(self, action):
        if action == 1:
            n_free_server = self.state[0] - 1
        else:
            n_free_server = self.state[0]
        priority = np.random.choice(self.priorities)
        self.state = (n_free_server, priority)
        return self.state

    def giveReward(self, action):
        # recieve a reward by taking the action
        if action == 1:
            priority = self.state[1]
            return np.power(2, priority)
        return 0

    def run(self, valueFunc, steps=1000, inner_steps=100, debug=False):
        # updating average reward estimation along the way
        avg_reward = 0
        self.state = (10, np.random.choice(self.priorities))
        cur_state = self.state
        cur_action = self.chooseAction(valueFunc)  # n free server is also updated

        total_reward = 0
        for i in range(1, steps + 1):
            reward = self.giveReward(cur_action)
            new_state = self.nxtState(cur_action)
            new_action = self.chooseAction(valueFunc)

            total_reward += reward
            if debug:
                print("state {} action {} reward {}".format(cur_state, cur_action, reward))
            if i % inner_steps == 0:
                print("step {} -> avg reward {} total reward {}".format(i, avg_reward, total_reward))

            #             target = reward - avg_reward + valueFunc.value(new_state, new_action)
            delta = reward - avg_reward + valueFunc.value(new_state, new_action) - valueFunc.value(cur_state,
                                                                                                   cur_action)
            avg_reward += self.beta * delta
            valueFunc.update(cur_state, cur_action, delta)

            cur_state = new_state
            cur_action = new_action


if __name__ == "__main__":
    sa = ServerAcess(exp_rate=0.1)
    vf = ValueFunction()
    sa.run(vf, steps=50000, inner_steps=5000, debug=False)

    plt.figure(figsize=[10, 6])

    for prioriy in range(4):
        n_servers = []
        values = []
        for n_server in range(11):
            value = vf.stateValue((n_server, prioriy))
            n_servers.append(n_server)
            values.append(value)
        plt.plot(n_servers, values, label="priority {}".format(np.power(2, prioriy)))
    plt.legend()
