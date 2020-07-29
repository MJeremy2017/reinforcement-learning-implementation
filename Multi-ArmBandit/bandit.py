import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, k=10, exp_rate=.3, lr=0.1, ucb=False, seed=None, c=2):
        self.k = k
        self.actions = range(self.k)
        self.exp_rate = exp_rate
        self.lr = lr
        self.total_reward = 0
        self.avg_reward = []

        self.TrueValue = []
        np.random.seed(seed)
        for i in range(self.k):
            self.TrueValue.append(np.random.randn() + 2)  # normal distribution

        self.values = np.zeros(self.k)
        self.times = 0
        self.action_times = np.zeros(self.k)

        self.ucb = ucb  # if select action using upper-confidence-bound
        self.c = c

    def chooseAction(self):
        # explore
        if self.ucb:
            if self.times == 0:
                action = np.random.choice(self.actions)
            else:   #  Even after a good value found, after many trials, UCB of this action might become lower than the lesser values, leading to retrying those again.
                confidence_bound = self.values + self.c * np.sqrt(
                    np.log(self.times) / (self.action_times + 0.1))  # c=2
                action = np.argmax(confidence_bound)
        else:   #  epsilon greedy
            if np.random.uniform(0, 1) <= self.exp_rate: # explore
                action = np.random.choice(self.actions)
            else:
                action = np.argmax(self.values) # exploit
        return action

    def takeAction(self, action):
        self.times += 1
        self.action_times[action] += 1
        # take action and update value estimates
        # reward = self.TrueValue[action]
        reward = np.random.randn() + self.TrueValue[action]  # add randomness to reward
        # using incremental method to propagate
        self.values[action] += self.lr * (reward - self.values[action])  # look like fixed lr converges better

        self.total_reward += reward
        self.avg_reward.append(self.total_reward / self.times)

    def play(self, n):
        for _ in range(n):
            action = self.chooseAction()
            self.takeAction(action)


if __name__ == "__main__":
    bdt = Bandit(k=5, exp_rate=0.1, seed=1234)
    bdt.play(2000)

    print("Estimated values", bdt.values)
    print("Actual values", bdt.TrueValue)
    avg_reward1 = bdt.avg_reward

    bdt = Bandit(k=5, exp_rate=0.3, seed=1234)
    bdt.play(2000)

    print("Estimated values", bdt.values)
    print("Actual values", bdt.TrueValue)

    avg_reward2 = bdt.avg_reward

    bdt = Bandit(k=5, exp_rate=0.1, seed=1234, ucb=True, c=2)
    bdt.play(2000)

    print("Estimated values", bdt.values)
    print("Actual values", bdt.TrueValue)

    avg_reward3 = bdt.avg_reward

    bdt = Bandit(k=5, exp_rate=0.1, seed=1234, ucb=True, c=5)
    bdt.play(2000)

    print("Estimated values", bdt.values)
    print("Actual values", bdt.TrueValue)

    avg_reward4 = bdt.avg_reward

    # reward plot
    plt.figure(figsize=[8, 6])
    plt.plot(avg_reward1, label="exp_rate=0.1")
    plt.plot(avg_reward2, label="exp_rate=0.3")
    plt.plot(avg_reward3, label="ucb, c=2")
    plt.plot(avg_reward4, label="ucb, c=5")

    plt.xlabel("n_iter", fontsize=14)
    plt.ylabel("avg reward", fontsize=14)
    plt.legend()
