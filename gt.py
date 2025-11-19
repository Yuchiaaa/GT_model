import numpy as np
import random
import matplotlib.pyplot as plt

from pyics import Model


def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""
    
    if n == 0:
        return [0]
    
    digits = []
    while n > 0:
        digits.append(n % k)
        n //= k
    
    return digits[::-1]

# A dictionary to hold different strategies for the experiments
strategies = {}

class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('rule', 30, setter=self.setter_rule)
        
        # I add a parameter here to choose between random initial condition or single seed
        self.make_param('random', False)

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def build_rule_set(self):
        """
        
        """
        
        return self.rule_set

    def check_rule(self, inp):
        """
        
        """
        
        
        return self.new_state

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""
        
        if self.random:
        # Generates a NumPy array of random integers in the range [0, self.k]
            return np.random.randint(
                    low=0, 
                    high=self.k, 
                    size=self.width,
                    dtype=int
            )
        else:
            # Initialize an array of zeros
            row = np.zeros(self.width, dtype=int)
            
            # Set the single non-zero value
            single_seed_value = 1
            
            # Calculate the center index
            center_index = self.width // 2 
            
            # Set the cell in the exact middle of the line to the seed value
            row[center_index] = single_seed_value
            
            return row

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                    for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)





def run_experiment(width=10, max_steps=100000, repeats=50, random=True):
    '''
    
    '''
    results = []
    errors = []


        
    print("Experiment completed.")
    return results, errors



if __name__ == '__main__':
    # Create the simulation model to be run in the GUI
    sim = CASim()

    from pyics import GUI
    cx = GUI(sim)
    cx.start()
    
    
    
