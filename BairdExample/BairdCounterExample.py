import numpy as np
import matplotlib.pyplot as plt

STATES = range(7)


class Baird:

    def __init__(self, gamma=0.99, alpha=0.01):
        self.state = np.random.choice(STATES)
        self.prob = 1.0 / 7
        self.actions = ["solid", "dash"]
        self.gamma = gamma
        self.alpha = alpha

        self.features = np.zeros((len(STATES), 8))  # n_states x n_weights (this is the representation of states)
        for i in range(len(STATES)):
            if i == 6:
                self.features[i, -2] = 1
                self.features[i, -1] = 2
            else:
                self.features[i, i] = 2
                self.features[i, -1] = 1

        self.weights = np.ones(8)
        self.weights[-2] = 10

    def chooseAction(self):
        if np.random.binomial(1, self.prob) == 1:
            action = "solid"
        else:
            action = "dash"
        return action

    def takeAction(self, action):
        if action == "solid":
            nxtState = 6
        else:
            nxtState = np.random.choice(STATES[:-1])
        return nxtState

    def value(self, state):
        v = np.dot(ba.features[state, :], ba.weights)
        return v

    def run_semi_gradient_TD(self, rounds=100, sarsa=False):
        reward = 0

        step_weights = np.zeros((rounds, len(self.weights)))
        for i in range(rounds):
            step_weights[i, :] = self.weights
            action = self.chooseAction()
            nxtState = self.takeAction(action)

            if action == "dash":
                rho = 0
            else:
                rho = 1 / self.prob

            if sarsa:
                rho = 1

            delta = reward + self.gamma * self.value(nxtState) - self.value(self.state)
            delta *= self.alpha * rho
            # update
            self.weights += delta * self.features[state, :]

            self.state = nxtState
        return step_weights

    def run_TDC(self, beta=0.01, rounds=100):
        reward = 0
        v = np.zeros(8)

        step_weights = np.zeros((rounds, len(self.weights)))
        for i in range(rounds):
            step_weights[i, :] = self.weights
            action = self.chooseAction()
            nxtState = self.takeAction(action)

            if action == "dash":
                rho = 0
            else:
                rho = 1 / self.prob

            delta = reward + self.gamma * self.value(nxtState) - self.value(self.state)
            self.weights += self.alpha * rho * (delta * self.features[self.state, :] -
                                                self.gamma * self.features[nxtState, :] * np.dot(
                        self.features[self.state, :], v))
            v += beta * rho * (delta - np.dot(v, self.features[self.state, :])) * self.features[self.state, :]

            self.state = nxtState
        print("last v \n", v)
        return step_weights


if __name__ == "__main__":
    ba = Baird()
    step_weigts_q = ba.run_semi_gradient_TD(rounds=1000)

    ba = Baird()
    step_weigts_sarsa = ba.run_semi_gradient_TD(rounds=1000, sarsa=True)

    plt.figure(figsize=[15, 6])

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    row, col = step_weigts_q.shape
    for i in range(col):
        ax1.plot(range(row), step_weigts_q[:, i], label="w{}".format(i + 1))
        ax2.plot(range(row), step_weigts_sarsa[:, i], label="w{}".format(i + 1))

    ax1.set_title("off-policy", size=14)
    ax2.set_title("on-policy", size=14)
    plt.legend()

    ba = Baird(alpha=0.005)
    step_weigts_tdc = ba.run_TDC(rounds=1000, beta=0.05)

    plt.figure(figsize=[10, 7])

    row, col = step_weigts_tdc.shape

    for i in range(col):
        plt.plot(range(row), step_weigts_q[:, i], label="w{}".format(i + 1))

    plt.xlabel("episode", size=14)
    plt.ylabel("value", size=14)
    plt.legend()