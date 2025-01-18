from .algorithm import Algorithm
import numpy as np
import scipy.stats


class SHAP_K(Algorithm):
    def __init__(self, t_min=0.5, delta=0.05):
        self.t_min = t_min
        self.delta = delta
    def sample(self, player):
        length = np.random.randint(self.game.n)
        sample = np.concatenate((np.arange(player), np.arange(player+1, self.game.n)))
        np.random.shuffle(sample)
        return sample[:length]
    def value(self, S: np.ndarray):
        length = S.shape[0]
        if length == 0:
            v = self.v_0
        elif length == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.T
            self.func_calls += 1
            v = self.game.value(S)
        return v
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        z_critical_value = scipy.stats.norm.ppf(1-(self.delta/n)/2)
        # print(z_critical_value)
        self.phi = np.zeros(n, dtype=np.float32)
        lower_bound = np.zeros(n, dtype=np.float32)
        upper_bound = np.zeros(n, dtype=np.float32)
        squared_marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        def update_player(player):
            S = self.sample(player)
            marginal = self.value(np.concatenate((S, [player]))) - self.value(S)
            t[player] += 1
            squared_marginals[player] += marginal**2
            self.phi[player] = ((t[player]-1)*self.phi[player] + marginal)/t[player]
            self.save_steps(step_interval)
            
        def update_bounds():
            sigma = np.sqrt((squared_marginals-t*(self.phi**2))/(t-1))
            c = z_critical_value*sigma/np.sqrt(t)
            nonlocal lower_bound
            nonlocal upper_bound
            lower_bound = self.phi - c
            upper_bound = self.phi + c
        for m in range(self.t_min):
            for player in range(n):
                update_player(player)
        update_bounds()
        while self.func_calls+4 <= self.T:
            sorted = np.argsort(-self.phi)
            topk, rest = sorted[:k], sorted[k:]
            h = topk[np.argmin(lower_bound[topk])]
            l = rest[np.argmax(upper_bound[rest])]
            
            update_player(h)
            update_player(l)
            
            update_bounds()
                
        self.func_calls = self.T
        self.save_steps(step_interval)