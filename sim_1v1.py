# Name:         Wendy van Wooning
# Studentnr.:   12493104
# Study:        BSc Informatica
#
# This program is used for

import numpy as np

from pyics import Model


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.config = None
        self.score_1 = 0
        self.score_2 = 0
        self.moves_1 = []
        self.moves_2 = []

        self.make_param('rounds', 10)
        self.make_param('strategy_1', 0, setter=self.setter_rule)
        self.make_param('strategy_2', 1, setter=self.setter_rule)

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        return max(0, min(val, 12))


    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each
        of the cells in the first row. Values should be between 0 and k."""
        cooperative_strats = [0, 2, 5, 6, 7, 8, 11]
        defensive_strats = [1, 3, 9]

        first_move_1 = np.random.randint(0, 2)
        first_move_2 = np.random.randint(0, 2)
        if self.strategy_1 in cooperative_strats:
            first_move_1 = 0
        elif self.strategy_1 in defensive_strats:
            first_move_1 = 1

        if self.strategy_2 in cooperative_strats:
            first_move_2 = 0
        elif self.strategy_1 in defensive_strats:
            first_move_2 = 1

        return [first_move_1, first_move_2]

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.score_1 = 0
        self.score_2 = 0
        self.moves_1 = []
        self.moves_2 = []
        self.config = np.zeros([self.rounds, 2])
        self.config[0, :] = self.setup_initial_row()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=1,
                   cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.rounds:
            return True

        move_p1 = self.play_strategy(0, 1, self.strategy_1)
        move_p2 = self.play_strategy(1, 0, self.strategy_2)
        self.moves_1.append(move_p1)
        self.moves_2.append(move_p2)

        self.config[self.t, 0] = move_p1
        self.config[self.t, 1] = move_p2

        self.calculate_score()


    def play_strategy(self, player, opponent, strat):
        # Set up: Get the previous round, and the list of all the player's and
        # the opponents played moves
        prev_round = [self.config[self.t - 1, 0], self.config[self.t - 1, 1]]
        if player == 0:
            own_moves = self.moves_1
            other_moves = self.moves_2
        else:
            own_moves = self.moves_2
            other_moves = self.moves_1

        # Horribly long list of all the strategies played
        # Strat 0 is the 'default', Tit-for-tat
        if strat == 0:
            # T-f-t always plays what the opponent previously played
            return prev_round[opponent]
        # 'T-f-t'-opposed, play the opposite of what t-f-t would play
        elif strat == 1:
            return (prev_round[opponent] + 1) % 2
        # Strat 2-4 are the simplest, always same move, or alternate
        elif strat == 2:
            return 0
        elif strat == 3:
            return 1
        elif strat == 4:
            return (prev_round[player] + 1) % 2
        # 'Vengeful', if the other player ever 'defects', now always defect
        elif strat == 5:
            if 1 in other_moves:
                return 1
            return 0
        # 'Forgiving', always cooperate if other player played that once
        elif strat == 6:
            if 0 in other_moves:
                return 0
            return 1
        # Try cooperation, if both players played the same move previously
        elif strat == 7:
            return (prev_round[player] + prev_round[opponent]) % 2
        # 'Calculating', only cooperate with a cooperative opponent
        elif strat == 8:
            print(len(other_moves))
            if len(other_moves) > 0:
                avg = sum(other_moves) / len(other_moves)
                print(avg)
                if avg > 0.5:
                    return 1
            return 0
        # 'Strategic', cooperate with very cooperative opponents, but only
        # those who sometimes defect
        elif strat == 9:
            if len(other_moves) > 0:
                avg = sum(other_moves) / len(other_moves)
                if avg < 0.5 and 1 in other_moves:
                    return 0
            return 1
        # 'Unpredictable' plays either randomly or their most unlikely move
        elif strat == 10:
            # Initially, play randomly
            if self.t < 10:
                return np.random.randint(0, 2)
            # Decide on unlikeliest move
            else:
                prev_moves = []
                for i in range(5):
                    prev_moves.append(own_moves[self.t - i])
                n_defects = sum(prev_moves)
                # Indistinct, just play randomly
                if n_defects == 2 or n_defects == 3:
                    return np.random.randint(0, 2)
                # Play the unlikelier move
                elif n_defects < 2:
                    return 1
                return 0
        # 'Forgiving t-f-t' tries to cooperate once in a while
        elif strat == 11:
            if self.t > 10:
                # Try to be forgiving, maybe both of you got stuck in a 'defect loop'
                if prev_round[player] == 1:
                    prev_moves = []
                    for i in range(5):
                        prev_moves.append(own_moves[self.t - 2 - i])
                    n_defects = sum(prev_moves)
                    if n_defects == 5:
                        return 0
            return prev_round[opponent]
        # Finally, the last 'strategy', just a random move
        return np.random.randint(0, 2)


    def calculate_score(self):
        # Player 1 cooperated
        if self.config[self.t, 0] == 0:
            # Both cooperate
            if self.config[self.t, 1] == 0:
                self.score_1 = self.score_1 + 3
                self.score_2 = self.score_2 + 3
            # Player 2 defected
            else:
                self.score_1 = self.score_1 + 0
                self.score_2 = self.score_2 + 5
        # Player 1 defected
        else:
            # Player 2 cooperated
            if self.config[self.t, 1] == 0:
                self.score_1 = self.score_1 + 5
                self.score_2 = self.score_2 + 0
            # Both defected
            else:
                self.score_1 = self.score_1 + 1
                self.score_2 = self.score_2 + 1

        if self.t == self.rounds - 1:
            print(f'Score player 1: {self.score_1}')
            print(f'Score player 1: {self.score_2}\n')


if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()

