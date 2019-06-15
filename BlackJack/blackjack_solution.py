import numpy as np
import pickle


class BlackJackSolution:

    def __init__(self, lr=0.1, exp_rate=0.3):
        self.player_Q_Values = {}  # key: [(player_value, show_card, usable_ace)][action] = value
        # initialise Q values | (12-21) x (1-10) x (True, False) x (1, 0) 400 in total
        for i in range(12, 22):
            for j in range(1, 11):
                for k in [True, False]:
                    self.player_Q_Values[(i, j, k)] = {}
                    for a in [1, 0]:
                        if (i == 21) and (a == 0):
                            self.player_Q_Values[(i, j, k)][a] = 1
                        else:
                            self.player_Q_Values[(i, j, k)][a] = 0

        self.player_state_action = []
        self.state = (0, 0, False)  # initial state
        self.actions = [1, 0]  # 1: HIT  0: STAND
        self.end = False
        self.lr = lr
        self.exp_rate = exp_rate

    # give card
    @staticmethod
    def giveCard():
        # 1 stands for ace
        c_list = list(range(1, 11)) + [10, 10, 10]
        return np.random.choice(c_list)

    def dealerPolicy(self, current_value, usable_ace, is_end):
        if current_value > 21:
            if usable_ace:
                current_value -= 10
                usable_ace = False
            else:
                return current_value, usable_ace, True
        # HIT17
        if current_value >= 17:
            return current_value, usable_ace, True
        else:
            card = self.giveCard()
            if card == 1:
                if current_value <= 10:
                    return current_value + 11, True, False
                return current_value + 1, usable_ace, False
            else:
                return current_value + card, usable_ace, False

    def chooseAction(self):
        # if current value <= 11, always hit
        current_value = self.state[0]
        if current_value <= 11:
            return 1

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        #             print("random action", action)
        else:
            # greedy action
            v = -999
            action = 0
            for a in self.player_Q_Values[self.state]:
                if self.player_Q_Values[self.state][a] > v:
                    action = a
                    v = self.player_Q_Values[self.state][a]
        #             print("greedy action", action)
        return action

    # one can only has 1 usable ace
    # return next state
    def playerNxtState(self, action):
        current_value = self.state[0]
        show_card = self.state[1]
        usable_ace = self.state[2]

        if action:
            # action hit
            card = self.giveCard()
            if card == 1:
                if current_value <= 10:
                    current_value += 11
                    usable_ace = True
                else:
                    current_value += 1
            else:
                current_value += card
        else:
            # action stand
            self.end = True
            return (current_value, show_card, usable_ace)

        if current_value > 21:
            if usable_ace:
                current_value -= 10
                usable_ace = False
            else:
                self.end = True
                return (current_value, show_card, usable_ace)

        return (current_value, show_card, usable_ace)

    def winner(self, player_value, dealer_value):
        # player 1 | draw 0 | dealer -1
        winner = 0
        if player_value > 21:
            if dealer_value > 21:
                # draw
                winner = 0
            else:
                winner = -1
        else:
            if dealer_value > 21:
                winner = 1
            else:
                if player_value < dealer_value:
                    winner = -1
                elif player_value > dealer_value:
                    winner = 1
                else:
                    # draw
                    winner = 0
        return winner

    def _giveCredit(self, player_value, dealer_value):
        reward = self.winner(player_value, dealer_value)
        # backpropagate reward
        for s in reversed(self.player_state_action):
            state, action = s[0], s[1]
            reward = self.player_Q_Values[state][action] + self.lr*(reward - self.player_Q_Values[state][action])
            self.player_Q_Values[state][action] = round(reward, 3)

    def reset(self):
        self.player_state_action = []
        self.state = (0, 0, False)  # initial state
        self.end = False

    def deal2cards(self, show=False):
        # return value after 2 cards and usable ace
        value, usable_ace = 0, False
        cards = [self.giveCard(), self.giveCard()]
        if 1 in cards:
            value = sum(cards) + 10
            usable_ace = True
        else:
            value = sum(cards)
            usable_ace = False

        if show:
            return value, usable_ace, cards[0]
        else:
            return value, usable_ace

    def play(self, rounds=1000):
        for i in range(rounds):
            if i % 1000 == 0:
                print("round", i)

            # give 2 cards
            dealer_value, d_usable_ace, show_card = self.deal2cards(show=True)
            player_value, p_usable_ace = self.deal2cards(show=False)

            self.state = (player_value, show_card, p_usable_ace)
            print("init", self.state)

            # judge winner after 2 cards
            if player_value == 21 or dealer_value == 21:
                # game end
                next
            else:
                while True:
                    action = self.chooseAction()  # state -> action
                    if self.state[0] >= 12:
                        state_action_pair = [self.state, action]
                        self.player_state_action.append(state_action_pair)
                    # update next state
                    self.state = self.playerNxtState(action)
                    if self.end:
                        break

                        # dealer's turn
                is_end = False
                while not is_end:
                    dealer_value, d_usable_ace, is_end = self.dealerPolicy(dealer_value, d_usable_ace, is_end)

                # judge winner
                # give reward and update Q value
                player_value = self.state[0]
                print("player value {} | dealer value {}".format(player_value, dealer_value))
                self._giveCredit(player_value, dealer_value)
  
            self.reset()

    def savePolicy(self, file="policy"):
        fw = open(file, 'wb')
        pickle.dump(self.player_Q_Values, fw)
        fw.close()

    def loadPolicy(self, file="policy"):
        fr = open(file, 'rb')
        self.player_Q_Values = pickle.load(fr)
        fr.close()

    # trained robot play against dealer
    def playWithDealer(self, rounds=1000):
        self.reset()
        self.loadPolicy()
        self.exp_rate = 0

        result = np.zeros(3)  # player [win, draw, lose]
        for _ in range(rounds):
            # hit 2 cards each
            # give 2 cards
            dealer_value, d_usable_ace, show_card = self.deal2cards(show=True)
            player_value, p_usable_ace = self.deal2cards(show=False)

            self.state = (player_value, show_card, p_usable_ace)

            # judge winner after 2 cards
            if player_value == 21 or dealer_value == 21:
                if player_value == dealer_value:
                    result[1] += 1
                elif player_value > dealer_value:
                    result[0] += 1
                else:
                    result[2] += 1
            else:
                # player's turn
                while True:
                    action = self.chooseAction()
                    # update next state
                    self.state = self.playerNxtState(action)
                    if self.end:
                        break

                        # dealer's turn
                is_end = False
                while not is_end:
                    dealer_value, d_usable_ace, is_end = self.dealerPolicy(dealer_value, d_usable_ace, is_end)

                # judge
                player_value = self.state[0]
                # print("player value {} | dealer value {}".format(player_value, dealer_value))
                w = self.winner(player_value, dealer_value)
                if w == 1:
                    result[0] += 1
                elif w == 0:
                    result[1] += 1
                else:
                    result[2] += 1
            self.reset()
        return result


if __name__ == "__main__":
    # training
    b = BlackJackSolution()
    b.play(10000)
    print("Done training")

    # save policy
    b.savePolicy()

    # play
    result = b.playWithDealer(rounds=1000)
    print(result)