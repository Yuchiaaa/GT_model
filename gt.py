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
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""
        rule_set_size = self.k ** (2 * self.r + 1)
        self.rule_set = [0] * rule_set_size
        base_k_rep = decimal_to_base_k(self.rule, self.k)
        for i in range(len(base_k_rep)):
            self.rule_set[-(i + 1)] = base_k_rep[-(i + 1)]
        return self.rule_set

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""
        
        length = 2*self.r + 1
        total = self.k ** length

        patterns = []
        for num in range(total):
            temp = num
            pattern = []
            for _ in range(length):
                pattern.append(temp % self.k)
                temp //= self.k
            patterns.append(pattern[::-1])
            
        patterns.sort(reverse=True)

        state_dict = {}
        for i in range(len(patterns)):
            state_dict[tuple(patterns[i])] = self.rule_set[i]

        integer_input_tuple = tuple(int(x) for x in inp)
        self.new_state = state_dict[integer_input_tuple]
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



def compute_cycle_length(sim, max_steps=100000):
    # initialize the simulation
    sim.reset()
    
    # dictionary to store seen row states and their time steps
    seen = {}
    
    for t in range(max_steps):
        # current row state
        row = tuple(sim.config[sim.t])  
        
        if row in seen:
            return t - seen[row]
        
        seen[row] = t
        finished = sim.step()
        if finished:
            break
    
    # if no cycle is found within max_steps, return max_steps
    return max_steps



wolfram_class = {
'0': 1, '1': 2, '2': 2, '3': 2, '4': 2, '5': 2, '6': 2, '7': 2, '8': 1, '9': 2, '10': 2, '11': 2, '12': 2, '13': 2, '14': 2, '15': 2, 
'16': 2, '17': 2, '18': 3, '19': 2, '20': 2, '21': 2, '22': 3, '23': 2, '24': 2, '25': 2, '26': 2, '27': 2, '28': 2, '29': 2, '30': 3, '31': 2, 
'32': 1, '33': 2, '34': 2, '35': 2, '36': 2, '37': 2, '38': 2, '39': 2, '40': 1, '41': 2, '42': 2, '43': 2, '44': 2, '45': 3, '46': 2, '47': 2, 
'48': 2, '49': 2, '50': 2, '51': 2, '52': 2, '53': 2, '54': 4, '55': 2, '56': 2, '57': 2, '58': 2, '59': 2, '60': 3, '61': 2, '62': 2, '63': 2, 
'64': 1, '65': 2, '66': 2, '67': 2, '68': 2, '69': 2, '70': 2, '71': 2, '72': 2, '73': 2, '74': 2, '75': 3, '76': 2, '77': 2, '78': 2, '79': 2, 
'80': 2, '81': 2, '82': 2, '83': 2, '84': 2, '85': 2, '86': 3, '87': 2, '88': 2, '89': 3, '90': 3, '91': 2, '92': 2, '93': 2, '94': 2, '95': 2, 
'96': 1, '97': 2, '98': 2, '99': 2, '100': 2, '101': 3, '102': 3, '103': 2, '104': 2, '105': 3, '106': 4, '107': 2, '108': 2, '109': 2, '110': 4, '111': 2, 
'112': 2, '113': 2, '114': 2, '115': 2, '116': 2, '117': 2, '118': 2, '119': 2, '120': 4, '121': 2, '122': 3, '123': 2, '124': 4, '125': 2, '126': 3, '127': 2, 
'128': 1, '129': 3, '130': 2, '131': 2, '132': 2, '133': 2, '134': 2, '135': 3, '136': 1, '137': 4, '138': 2, '139': 2, '140': 2, '141': 2, '142': 2, '143': 2, 
'144': 2, '145': 2, '146': 3, '147': 4, '148': 2, '149': 3, '150': 3, '151': 3, '152': 2, '153': 3, '154': 2, '155': 2, '156': 2, '157': 2, '158': 2, '159': 2, 
'160': 1, '161': 3, '162': 2, '163': 2, '164': 2, '165': 3, '166': 2, '167': 2, '168': 1, '169': 4, '170': 2, '171': 2, '172': 2, '173': 2, '174': 2, '175': 2, 
'176': 2, '177': 2, '178': 2, '179': 2, '180': 2, '181': 2, '182': 3, '183': 3, '184': 2, '185': 2, '186': 2, '187': 2, '188': 2, '189': 2, '190': 2, '191': 2, 
'192': 1, '193': 4, '194': 2, '195': 3, '196': 2, '197': 2, '198': 2, '199': 2, '200': 2, '201': 2, '202': 2, '203': 2, '204': 2, '205': 2, '206': 2, '207': 2, 
'208': 2, '209': 2, '210': 2, '211': 2, '212': 2, '213': 2, '214': 2, '215': 2, '216': 2, '217': 2, '218': 2, '219': 2, '220': 2, '221': 2, '222': 2, '223': 2, 
'224': 1, '225': 4, '226': 2, '227': 2, '228': 2, '229': 2, '230': 2, '231': 2, '232': 2, '233': 2, '234': 1, '235': 1, '236': 2, '237': 2, '238': 1, '239': 1, 
'240': 2, '241': 2, '242': 2, '243': 2, '244': 2, '245': 2, '246': 2, '247': 2, '248': 1, '249': 1, '250': 1, '251': 1, '252': 1, '253': 1, '254': 1, '255': 1
}

wolfram_rule = list(range(256))

def get_wolfram_class(rule_number):
    '''I return the wolfram class of a given rule number as an integer between 1 and 4.'''
    return wolfram_class[str(rule_number)]




def run_experiment(width=10, max_steps=100000, repeats=50, random=True):
    '''I run an experiment to compute the average cycle length for each of the 256
    wolfram rules. Each rule is repeated `repeats` times to get an average cycle length.
    each rule starts with a random initial condition if `random` is True. by default, 
    I would like to use different initial conditions for each rule in order to get a general answer.'''
    results = []
    errors = []

    for r in range(256):
        cycle_lengths = []
        for _ in range(repeats):
            sim = CASim()
            sim.width = width
            sim.height = max_steps
            sim.random = random
            sim.rule = r
            cycle_length = compute_cycle_length(sim, max_steps)
            cycle_lengths.append(cycle_length)
        
        avg_cycle = np.mean(cycle_lengths)
        std_err = np.std(cycle_lengths) / np.sqrt(repeats)
        results.append(avg_cycle)
        errors.append(std_err)
        
    print("Experiment completed.")
    return results, errors




def extra_experiment(width=10, max_steps=100000, repeats = 50, rule = 30, random =True, random_bases=range(2, 11)):
    '''I want to try if rule 30, which is in class one, always keeps a low complexity on any bases.
    each base is repeated `repeats` times to get an average cycle length.
    each run starts with a random initial condition if `random` is True. by default,'''
    results = []
    errors = []
    
    for base in random_bases:
        cycle_lengths = []
        for _ in range(repeats):
            sim = CASim()
            sim.width = width
            sim.height = max_steps
            sim.rule = rule
            sim.k = base
            sim.random = random

            cycle_length = compute_cycle_length(sim, max_steps)
            cycle_lengths.append(cycle_length)
        
        avg_cycle = np.mean(cycle_lengths)
        std_err = np.std(cycle_lengths) / np.sqrt(repeats)
        results.append(avg_cycle)
        errors.append(std_err)
        
    print("Extra experiment completed.")
    return results, errors
    

if __name__ == '__main__':
    sim = CASim() # create a simulation instance, make sure to include this line before running the experiments.
    # steps to visualize a certain rule using GUI (to test if the simulation works fine), unnecessary for the experiments
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
    
    # starting parameters for the experiment
    width = 10 # width of the CA
    time_limit = 100000 # maximum steps to search for cycles
    repeats = 50 # number of repeats for each rule
    random = True # use random initial conditions
    
    # run the experiment
    results, errors = run_experiment(width=width, max_steps=time_limit, random=random) # although I calculate errors, I won't plot them as error bars later, considering the amount of bars in the picture.


    # graph_1: plotting the results （classified by wolfram class)
    wolfram_class_int_keys = {int(k): v for k, v in wolfram_class.items()}
    wolfram_rule.sort(key=lambda rule: (wolfram_class_int_keys[rule], rule))
    sorted_results = [results[rule] for rule in wolfram_rule]
    
    # transform wolfram_rule to string because I will change its original order I don't want system to automatically sort it again
    wolfram_rule_str = [str(r) for r in wolfram_rule]
    
    # get the wolfram classes for coloring
    classes = [wolfram_class_int_keys[r] for r in wolfram_rule]
    
    # assign colors based on wolfram classes
    unique_classes = []
    for c in classes:
        if c not in unique_classes:
            unique_classes.append(c)
        
    colors_list = ['blue', 'green', 'orange', 'red']
    class_to_color = {cls: colors_list[i % len(colors_list)] for i, cls in enumerate(unique_classes)}
    
    bar_colors = [class_to_color[c] for c in classes]
    
    # plotting the results （ordered by wolfram class)
    plt.bar(wolfram_rule_str, sorted_results, color=bar_colors)
    plt.xlabel("Class 1 (Rules 0,8,32,...) --> Class 2 --> Class 3 --> Class 4 (Rule 54,106,110,120)")
    
    # remove x ticks for better visualization
    plt.xticks([])
    plt.ylabel("Average Cycle Length")
    plt.text(0.95, 0.95,
         f"r={sim.r}, k={sim.k}, time_limit={time_limit}, width={width}, random_seed={random}",
         ha='right', va='top', transform=plt.gca().transAxes)
    plt.title("Cycle Length of Wolfram Rules (Classified by Wolfram Class)")
    plt.show()
    # after running the above code, you will get the picture "classified_cycle_length.png" in the submission folder.
    
    
    
    # extra experiment:
    # I want to test how does the complexity of a rule in class 3 perform on different bases
    # let us try rule 30, which is in class 3
    width = 10
    time_limit = 100000
    repeats = 50
    random = True
    bases = range(2, 11)
    test_r = 30
    
    # run the extra experiment
    extra_results, extra_errors = extra_experiment(width=width, max_steps=time_limit, repeats=repeats, rule=test_r, random=random, random_bases=bases)
    
    # graph_2: plotting the extra experiment results
    plt.bar(bases, extra_results, yerr=extra_errors, color='purple')
    plt.xlabel("Base k (from 2 to 10)")
    plt.ylabel("Average Cycle Length")
    plt.text(0.95, 0.95,
         f"r={sim.r}, k={sim.k}, time_limit={time_limit}, width={width}, random_seed={random}",
         ha='right', va='top', transform=plt.gca().transAxes)
    plt.title(f"Cycle Length of Rule {test_r} on Different Bases")
    plt.show()
    # after running the above code, you will get the picture "extra.png" in the submission folder.
    # please notice that we can't see the error bars in some bars since their value is 0 which says that the system behaves very stably under these parameters, with the period length remaining almost constant.
    
    

