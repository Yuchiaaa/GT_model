import numpy as np
import random
import matplotlib.pyplot as plt

from pyics import Model


strategies = {
    "TitForTat": [0, 0, 1, 0, 1],
    "AllC": [0, 0, 0, 0, 0],    
    "AllD": [1, 1, 1, 1, 1],     
    "Random": [random.randint(0, 1) for _ in range(5)], 
    "s1": [0, 1, 0, 1, 0], 
    "s2": [1, 0, 1, 0, 1], 
    # Add more original strategies
}

def get_next_move(strategy_gene, own_history, opp_history):
    """
    based on strategy_gene and history, return next move (0=C, 1=D)
    """
    
    # initial move
    if not own_history:
        return strategy_gene[0]
    
    # get last moves
    own_prev = own_history[-1]
    opp_prev = opp_history[-1]

    history_index = own_prev * 2 + opp_prev + 1
    
    return strategy_gene[history_index]

def RunRepeatedPrisonerDilemma(strategy_a, strategy_b, repetitions, payoff_table):
    """
    run repeated prisoner's dilemma between two strategies
    """
    
    a_history = []
    b_history = []
    a_score = 0
    b_score = 0
    
    for _ in range(repetitions):
        # make choices
        a_choice = get_next_move(strategy_a, a_history, b_history)
        b_choice = get_next_move(strategy_b, b_history, a_history)
        
        # get payoffs
        score_a = payoff_table[(a_choice, b_choice)]
        score_b = payoff_table[(b_choice, a_choice)]
        
        a_score += score_a
        b_score += score_b
        
        # update history
        a_history.append(a_choice)
        b_history.append(b_choice)
        
    return a_score, b_score

class EvolutionSim(Model):
    def __init__(self):
        Model.__init__(self)

        self.generation = 0
        self.population = []
        self.history_scores = []

        self.make_param('N', 50) 
        
        self.make_param('T', 200)
        
        self.make_param('MaxGenerations', 50)
        
        self.make_param('MutationRate', 0.05) 
        self.make_param('CrossoverRate', 0.8) 
        self.make_param('ElitismCount', 5)
        
        # payoff table for Prisoner's Dilemma
        self.payoff_table = {(0, 0): 3, (0, 1): 0, (1, 0): 5, (1, 1): 1}

    def setup_initial_population(self):
        """
        initialize population with predefined strategies and random strategies
        """
        
        gene_length = 5 
        initial_pop = []
        
        # cite predefined strategies
        predefined_keys = list(strategies.keys())
        
        for key in predefined_keys:
            if key != "Random":
                initial_pop.append(strategies[key])
                
        remaining = self.N - len(initial_pop)
        for _ in range(remaining):
            random_gene = [random.randint(0, 1) for _ in range(gene_length)]
            initial_pop.append(random_gene)
            
        return initial_pop

    def reset(self):
        self.generation = 0
        self.population = self.setup_initial_population()
        self.history_scores = []

    def run_tournament(self):
        """
        run a global tournament among all strategies in the population
        """ 
        
        N = self.N

        strategies_with_fitness = []
        
        for i in range(N):
            strategies_with_fitness.append({'gene': self.population[i], 'fitness': 0})
            
        for i in range(N):
            for j in range(i + 1, N):
                
                strategy_i = strategies_with_fitness[i]['gene']
                strategy_j = strategies_with_fitness[j]['gene']
                
                score_i, score_j = RunRepeatedPrisonerDilemma(
                    strategy_i, 
                    strategy_j, 
                    self.T, 
                    self.payoff_table
                )
                
                strategies_with_fitness[i]['fitness'] += score_i
                strategies_with_fitness[j]['fitness'] += score_j
                
        return strategies_with_fitness

    def step(self):
        """
        perform one generation step
        """
        
        if self.generation >= self.MaxGenerations:
            return True

        # run tournament to get scored population
        scored_population = self.run_tournament()
        
        # record average fitness
        total_fitness = sum(s['fitness'] for s in scored_population)
        avg_fitness = total_fitness / self.N
        self.history_scores.append(avg_fitness)
        
        # generate new population using genetic algorithm
        self.population = genetic_algorithm(
            scored_population,
            self.N,
            self.ElitismCount,
            self.CrossoverRate,
            self.MutationRate
        )
        
        self.generation += 1

    def draw(self):
        """
        draw the evolution of average fitness over generations
        """
        plt.cla()
        

        plt.plot(self.history_scores, label='Average Population Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Average Total Score (Fitness)')
        plt.title('Evolution of Cooperation (Generation %d)' % self.generation)
        plt.legend()
        
def select_parent(scored_population):
    """
    based on fitness, select a parent strategy
    """

    total_fitness = sum(s['fitness'] for s in scored_population)
    if total_fitness == 0:
        return random.choice(scored_population)['gene']
        
    r = random.uniform(0, total_fitness)
    current_sum = 0
    for individual in scored_population:
        current_sum += individual['fitness']
        if current_sum > r:
            return individual['gene']
            
    # Fallback to the last one if floating point precision issues occur
    return scored_population[-1]['gene']

def crossover(parent1_gene, parent2_gene):
    """
    single-point crossover between two parent genes
    """
    gene_length = len(parent1_gene)
    point = random.randint(1, gene_length - 1)
    
    child1 = parent1_gene[:point] + parent2_gene[point:]
    child2 = parent2_gene[:point] + parent1_gene[point:]
    
    return child1, child2

def mutate(gene, rate):
    """
    mutate the gene with given mutation rate
    """
    mutated_gene = list(gene)
    for i in range(len(mutated_gene)):
        if random.random() < rate:
            mutated_gene[i] = 1 - mutated_gene[i] # 0 -> 1, 1 -> 0
    return mutated_gene

def genetic_algorithm(scored_population, N, elitism_count, crossover_rate, mutation_rate):
    """
    based on scored_population (list of dicts with 'gene' and 'fitness'),
    """
    
    # rank individuals by fitness
    scored_population.sort(key=lambda x: x['fitness'], reverse=True)
    
    new_population = []
    
    # retain the best individuals
    for i in range(min(elitism_count, N)):
        new_population.append(scored_population[i]['gene'])
        
    # 3. fill the rest of the population
    while len(new_population) < N:
        parent1 = select_parent(scored_population)
        parent2 = select_parent(scored_population)
        
        # crossover
        if random.random() < crossover_rate and len(new_population) <= N - 2:
            child1_gene, child2_gene = crossover(parent1, parent2)
            
            # mutation
            new_population.append(mutate(child1_gene, mutation_rate))
            new_population.append(mutate(child2_gene, mutation_rate))
        else:
            # directly mutate parents if no crossover
            new_population.append(mutate(parent1, mutation_rate))
            
    # make sure we only return N individuals
    return new_population[:N]



if __name__ == '__main__':
    # Create the simulation model to be run in the GUI
    sim = EvolutionSim()

    from pyics import GUI
    cx = GUI(sim)
    cx.start()
    
    
    
