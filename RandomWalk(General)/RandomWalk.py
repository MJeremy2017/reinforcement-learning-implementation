import numpy as np
import matplotlib.pyplot as plt

NUM_STATES = 1000
START = 500
END_0 = 0
END_1 = 1001

TRUE_VALUES = np.arange(-1001, 1003, 2) / 1001.0


class LinearValueFunction:

    def __init__(self, order, method="poly"):
        if method == "poly":
            self.func = [lambda x, i=i: np.power(x, i) for i in range(0, order + 1)]  # s^i
        if method == "fourier":
            self.func = [lambda x, i=i: np.cos(np.pi * x * i) for i in range(0, order + 1)]  # cos(pi*s*i)
        self.weights = np.zeros(order + 1)

    def value(self, state):
        state = state / NUM_STATES
        features = np.array([f(state) for f in self.func])
        return np.dot(features, self.weights)

    def update(self, delta, state):
        state = state / NUM_STATES
        dev = np.array([f(state) for f in self.func])
        self.weights += delta * dev


class AggValueFunction:

    def __init__(self, num_groups=10):
        self.num_groups = num_groups
        self.group_size = NUM_STATES // self.num_groups
        self.values = np.zeros(self.num_groups)

    def value(self, state):
        # explicitly set end state value
        if state == END_0:
            return -1
        if state == END_1:
            return 1
        group = (state - 1) // self.group_size
        value = self.values[group]
        return value

    def update(self, delta, state):
        dev = 1  # derivative is 1 in this case
        group = (state - 1) // self.group_size
        self.values[group] += delta * dev


class RandomWalk:

    def __init__(self, step=1, lr=2e-5, gamma=1, debug=True):
        self.state = START
        self.actions = ["left", "right"]
        self.end = False
        self.n = step
        self.lr = lr
        self.gamma = gamma
        self.debug = debug

    def chooseAction(self):
        action = np.random.choice(self.actions)
        return action

    def takeAction(self, action):
        # choose steps from 1 to 100
        steps = np.random.choice(range(1, 101))
        if action == "left":
            state = self.state - steps
        else:
            state = self.state + steps
        # judge if end of game
        if state <= END_0 or state >= END_1:
            self.end = True
            if state <= END_0:
                state = END_0
            else:
                state = END_1

        self.state = state
        return state

    def giveReward(self):
        if self.state == END_0:
            return -1
        if self.state == END_1:
            return 1
        return 0

    def reset(self):
        self.state = START
        self.end = False

    def play(self, valueFunction, rounds=1e5):
        for rnd in range(rounds):
            self.reset()
            t = 0
            T = np.inf
            action = self.chooseAction()

            actions = [action]
            states = [self.state]
            rewards = [0]
            while True:
                if t < T:
                    state = self.takeAction(action)  # next state
                    reward = self.giveReward()  # next state-reward

                    states.append(state)
                    rewards.append(reward)

                    if self.end:
                        if self.debug:
                            if (rnd + 1) % 5000 == 0:
                                print("Round {}: End at state {} | number of states {}".format(rnd + 1, state,
                                                                                               len(states)))
                        T = t + 1
                    else:
                        action = self.chooseAction()
                        actions.append(action)  # next action
                # state tau being updated
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n + 1, T + 1)):
                        G += np.power(self.gamma, i - tau - 1) * rewards[i]
                    if tau + self.n < T:
                        state = states[tau + self.n]
                        G += np.power(self.gamma, self.n) * valueFunction.value(state)
                    # update value function
                    state = states[tau]
                    delta = self.lr * (G - valueFunction.value(state))
                    valueFunction.update(delta, state)

                if tau == T - 1:
                    break

                t += 1


def plot_fig(func):
    x_value = range(0, 1002)
    states_value = []
    for i in x_value:
        states_value.append(func.value(i))

    plt.figure(figsize=[8, 6])
    plt.plot(x_value, states_value, label="Approximate Value")
    plt.plot(x_value, TRUE_VALUES, label="Actual Value")
    plt.legend()


if __name__ == "__main__":
    # agg function
    print("Running agg function")
    rw = RandomWalk(step=1, lr=0.001)
    vFunc = AggValueFunction(num_groups=10)

    rw.play(rounds=5000, valueFunction=vFunc)

    plot_fig(vFunc)

    # poly function
    print("Running poly function")
    rw = RandomWalk(step=1, lr=0.001)
    polyFunc = LinearValueFunction(order=5, method="poly")
    rw.play(rounds=5000, valueFunction=polyFunc)

    plot_fig(polyFunc)

    # fourier function
    print("Running fourier function")
    rw = RandomWalk(step=1, lr=0.001)
    fFunc = LinearValueFunction(order=5, method="fourier")

    rw.play(rounds=5000, valueFunction=fFunc)
    plot_fig(fFunc)
