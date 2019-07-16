import numpy as np
import matplotlib.pyplot as plt
from TileCoding import *
from mpl_toolkits.mplot3d import Axes3D

VELOCITY_BOUND = [-0.07, 0.07]
POSITION_BOUND = [-1.2, 0.5]
ACTIONS = [-1, 0, 1]


class ValueFunction:

    def __init__(self, stepSize, numOfTilings=8, maxSize=2048):
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings

        # divide step size equally to each tiling
        self.stepSize = stepSize / numOfTilings  # learning rate for each tile

        self.hashTable = IHT(maxSize)

        # weight for each tile
        self.weights = np.zeros(maxSize)

        # position and velocity needs scaling to satisfy the tile software
        self.positionScale = self.numOfTilings / (POSITION_BOUND[1] - POSITION_BOUND[0])
        self.velocityScale = self.numOfTilings / (VELOCITY_BOUND[1] - VELOCITY_BOUND[0])

    # get indices of active tiles for given state and action
    def getActiveTiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        activeTiles = tiles(self.hashTable, self.numOfTilings,
                            [self.positionScale * position, self.velocityScale * velocity],
                            [action])
        return activeTiles

    # estimate the value of given state and action
    def value(self, position, velocity, action):
        if position == POSITION_BOUND[1]:
            return 0.0
        activeTiles = self.getActiveTiles(position, velocity, action)
        return np.sum(self.weights[activeTiles])

    # learn with given state, action and target
    def update(self, position, velocity, action, target):
        activeTiles = self.getActiveTiles(position, velocity, action)
        estimation = np.sum(self.weights[activeTiles])
        delta = self.stepSize * (target - estimation)
        for activeTile in activeTiles:
            self.weights[activeTile] += delta

    # get the # of steps to reach the goal under current state value function
    def costToGo(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)


class MountainCar:

    def __init__(self, n=1, exp_rate=0.1, gamma=1, debug=True):
        self.actions = [-1, 0, 1]  # reverse, 0 and forward throttle
        self.state = (-0.5, 0)  # position, velocity
        self.exp_rate = exp_rate
        self.gamma = gamma
        self.end = False
        self.n = n  # step of learning
        self.debug = debug

    def reset(self):
        pos = np.random.uniform(-0.6, -0.4)
        self.end = False
        self.state = (pos, 0)

    def takeAction(self, action):
        pos, vel = self.state

        vel_new = vel + 0.001 * action - 0.0025 * np.cos(3 * pos)
        vel_new = min(max(vel_new, VELOCITY_BOUND[0]), VELOCITY_BOUND[1])

        pos_new = pos + vel_new
        pos_new = min(max(pos_new, POSITION_BOUND[0]), POSITION_BOUND[1])

        if pos_new == POSITION_BOUND[0]:
            # reach leftmost, set speed to 0
            vel_new = 0
        self.state = (pos_new, vel_new)
        return self.state

    def chooseAction(self, valueFunc):
        # choose an action based on the current state,
        if np.random.uniform(0, 1) <= self.exp_rate:
            # random action
            return np.random.choice(self.actions)
        else:
            # greedy action
            values = {}
            for a in self.actions:
                pos, vel = self.state
                value = valueFunc.value(pos, vel, a)
                values[a] = value
            return np.random.choice([k for k, v in values.items() if v == max(values.values())])

    def giveReward(self):
        pos, _ = self.state
        if pos == POSITION_BOUND[1]:
            self.end = True
            return 0
        return -1

    def play(self, valueFunction, rounds=1):
        for rnd in range(1, rounds + 1):
            self.reset()
            t = 0
            T = np.inf
            action = self.chooseAction(valueFunction)

            actions = [action]
            states = [self.state]
            rewards = [-1]
            while True:
                if t < T:
                    state = self.takeAction(action)  # next state
                    reward = self.giveReward()  # next state-reward

                    states.append(state)
                    rewards.append(reward)

                    if self.end:
                        if self.debug:
                            if rnd % 500 == 0:
                                print("Round {}: End at state {} | number of states {}".format(rnd, state, len(states)))
                        T = t + 1
                    else:
                        action = self.chooseAction(valueFunction)
                        actions.append(action)  # next action
                # state tau being updated
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n + 1, T + 1)):
                        G += np.power(self.gamma, i - tau - 1) * rewards[i]
                    if tau + self.n < T:
                        state = states[tau + self.n]
                        G += np.power(self.gamma, self.n) * valueFunction.value(state[0], state[1],
                                                                                actions[tau + self.n])
                    # update value function
                    state = states[tau]  # tau is the state to update
                    valueFunction.update(state[0], state[1], actions[tau], G)

                if tau == T - 1:
                    break

                t += 1


if __name__ == "__main__":
    stepSize = 0.3
    valueFunc = ValueFunction(stepSize)
    mc = MountainCar()
    mc.play(valueFunc, rounds=100)

    grids = 50

    positions = np.linspace(POSITION_BOUND[0], POSITION_BOUND[1], grids)
    vels = np.linspace(VELOCITY_BOUND[0], VELOCITY_BOUND[1], grids)

    x = []
    y = []
    z = []
    for p in positions:
        for v in vels:
            x.append(p)
            y.append(v)
            z.append(valueFunc.costToGo(p, v))

    fig = plt.figure(figsize=[10, 6])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Cost to go")
