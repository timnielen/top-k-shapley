import numpy as np
import scipy.stats

class Algorithm:
    def initialize(self, game, budget: int, step_interval: int = 100):
        self.values = []
        self.game = game
        self.budget = budget # if budget == -1, the budget is unlimited and the algorithm needs to stop itself
        self.func_calls = 0
        n = game.n
        self.phi = np.zeros(n, dtype=np.float32)
        self.t = np.zeros(n, dtype=np.float32)
        self.step_interval = step_interval
        self.eval_empty_full()

    def get_top_k(self, k: int):
        '''
        evaluated the topk players. 
        The result is an approximation for every players shapley value stored in self.phi 
        -> can be used for the simple approximation problem aswell
        The intermediate values are stored in self.values according to self.step_interval.
        '''
        pass

    def save_steps(self, final=False):
        '''saves the intermediate estimates for the shapley values'''
        # in case of final = True, i.e. the final save in the end, we save the current values even if total budget is not actually reached. Is neccessary e.g. if an algorithm terminates with func_calls=budget-1
        if final and self.budget != -1:
            self.func_calls = self.budget
        if(self.func_calls/self.step_interval >= len(self.values) + 1):
            self.values += [np.array(self.phi)]
    
    def value(self, coalition: np.ndarray):
        # returns the value of a coalition (uses cached values if possible)
        length = coalition.shape[0]
        if length == 0:
            v = self.v_0
        elif length == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.budget or self.budget == -1
            self.func_calls += 1
            v = self.game.value(coalition)
        return v
        
    def eval_empty_full(self):
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(self.game.n))
        self.v_0 = self.game.value(np.array([]))
        
    def partition(self, k):
        # returns the current topk and non-topk players
        sorted = np.argsort(-self.phi)
        return sorted[:k], sorted[k:]
    
    def update_player(self, player, marginal):
        # utility function to update a players estimator using a sampled marginal contribution
        self.phi[player] = (self.t[player] * self.phi[player] + marginal)/(self.t[player] + 1)
        self.t[player] += 1
        self.save_steps(final=False)

    def sample(self, player: int):
        # utility function to sample a coalition S in N \ {player}
        n = self.game.n
        length = np.random.randint(n)
        S = np.concatenate((np.arange(player), np.arange(player+1, n)))
        np.random.shuffle(S) 
        return S[:length]
    
class PAC_Algorithm(Algorithm):
    def __init__(self, t_min=30, delta=0.01, epsilon=0.001):
        self.t_min = t_min
        self.delta = delta
        self.epsilon = epsilon
        
    def initialize(self, game, budget: int, step_interval: int = 100):
        super(PAC_Algorithm, self).initialize(game, budget, step_interval)
        n = game.n
        self.squared_marginals = np.zeros(n, dtype=np.float32)
        self.z_critical_value = scipy.stats.norm.ppf(1-(self.delta/n)/2)
        self.lower_bound = np.zeros(n, dtype=np.float32)
        self.upper_bound = np.ones(n, dtype=np.float32)
        self.topk_low = 0
        self.rest_high = 2*self.epsilon
    
    def update_player(self, player, marginal):
        # utility function to update a players estimator using a sampled marginal contribution
        self.phi[player] = (self.t[player] * self.phi[player] + marginal)/(self.t[player] + 1)
        self.squared_marginals[player] += marginal**2
        self.t[player] += 1
        self.save_steps(self.step_interval)
        
    def update_bounds(self, topk, rest):
        # update the confidence bounds of all players using the central limit theorem
        if np.any(self.t<self.t_min): # only update if t_min is reached, i.e. CLT is valid
            return
        sigma = np.sqrt((self.squared_marginals-self.t*(self.phi**2))/(self.t-1))
        c = self.z_critical_value*sigma/np.sqrt(self.t)
        self.lower_bound = self.phi - c
        self.upper_bound = self.phi + c
        self.topk_low = np.min(self.lower_bound[topk]) # smallest topk bound
        self.rest_high = np.max(self.upper_bound[rest]) # largest non-topk bound
            
    def is_PAC(self):
        # if the maximum overlap of the bounds of the two partitions is at most epsilon, the solution is probably approximately correct
        return self.rest_high - self.topk_low <= self.epsilon
    
    
 