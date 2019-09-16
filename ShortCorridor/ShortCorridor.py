import numpy as np


class ShortCorridor:
    def __init__(self, alpha=0.2, gamma=0.8):
        self.actions = ["left", "right"]
        self.x = np.array([[0, 1], [1, 0]])  # left|s, right|s
        self.theta = np.array([-1.47, 1.47])
        self.state = 0  # initial state 0
        self.gamma = gamma
        self.alpha = alpha

    def softmax(self, vector):
        return np.exp(vector) / sum(np.exp(vector))

    def chooseAction(self):
        h = np.dot(self.theta, self.x)
        prob = self.softmax(h)  # left, right probability for all state

        action = np.random.choice(self.actions, p=prob)
        return action, prob

    def takeAction(self, action):
        if self.state == 0:
            nxtState = 0 if action == "left" else 1
        elif self.state == 1:
            nxtState = 2 if action == "left" else 0  # reversed
        elif self.state == 2:
            nxtState = 1 if action == "left" else 3
        else:
            nxtState = 2 if action == "left" else 3
        return nxtState

    def giveReward(self):
        if self.state == 3:
            return 0
        return -1

    def reset(self):
        self.state = 0

    def run(self, rounds=100):
        actions = []
        rewards = []
        for i in range(1, rounds + 1):
            reward_sum = 0
            while True:
                action, prob = self.chooseAction()
                nxtState = self.takeAction(action)
                reward = self.giveReward()
                reward_sum += reward

                actions.append(action)
                rewards.append(reward)

                self.state = nxtState
                # game end
                if self.state == 3:
                    T = len(rewards)
                    for t in range(T):
                        # calculate G
                        G = 0
                        for k in range(t + 1, T):
                            G += np.power(self.gamma, k - t - 1) * rewards[k]

                        j = 1 if actions[t] == "right" else 0  # dev on particular state
                        h = np.dot(self.theta, self.x)
                        prob = self.softmax(h)
                        grad = self.x[:, j] - np.dot(self.x, prob)

                        self.theta += self.alpha * np.power(self.gamma, t) * G * grad
                    # reset
                    self.state = 0
                    actions = []
                    rewards = []

                    if i % 50 == 0:
                        print("round {}: current prob {} reward {}".format(i, prob, reward_sum))
                        reward_sum = 0
                    break


if __name__ == "__main__":
    sc = ShortCorridor(alpha=2e-4, gamma=1)
    sc.run(1000)

    h = np.dot(sc.theta, sc.x)
    print(sc.softmax(h))  # left, right probability for all state
