from .algorithm import Algorithm
import math
import numpy as np
import util
from scipy.stats import norm

class CMCS(Algorithm):
    def value(self, S: np.ndarray):
        l = S.shape[0]
        if l == 0:
            v = self.v_0
        elif l == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.T
            self.func_calls += 1
            v = self.game.value(S)
        return v
    def sample(self, length=None):
        if length is None:
            length = np.random.randint(self.game.n+1)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length]
    def marginal(self, coalition, player, value_coalition=None):
        if value_coalition is None:
            value_coalition = self.value(coalition)
        if player in coalition:
            return value_coalition - self.value(coalition[coalition != player])
        return self.value(np.concatenate((coalition, [player]))) - value_coalition
    def update_phi(self, new_value):
        self.phi = new_value
        self.save_steps(self.step_interval)
        
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+2 <= self.T:
            S = self.sample()
            v_S = self.value(S)
            for player in range(n):
                if self.func_calls == self.T:
                    self.update_phi(marginals / t)
                    return
                marginals[player] += self.marginal(S, player, v_S)
                t[player] += 1
            self.update_phi(marginals / t)
        self.func_calls = self.T
        self.save_steps(step_interval)

class Selective_CMCS(CMCS):
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+2 <= self.T:            
            # sample players based on uncertainty
            sorted_players = np.argsort(-self.phi)
            # border = np.zeros(n, dtype=np.float32)
            # border[sorted_players[:k]] = self.phi[sorted_players[k]]
            # border[sorted_players[k:]] = self.phi[sorted_players[k-1]]
            border = (self.phi[sorted_players[k-1]] + self.phi[sorted_players[k]])/2
            certainty = np.abs(self.phi - border) * t
            min_certainty, max_certainty = np.min(certainty), np.max(certainty)
            if min_certainty == max_certainty:
                selected_players = np.arange(n)
            else:
                weights = (max_certainty - certainty) / (max_certainty - min_certainty)
                selected_players = np.array((np.random.rand(n) < weights).nonzero())[0]
                
                # if no players where selected, update all to avoid a loop where only a little number of players is selected every time which is not budget efficient 
                if selected_players.shape[0] == 0:
                    selected_players = np.arange(n)
                        
            S = self.sample()
            v_S = self.value(S)
            for player in selected_players:
                if self.func_calls == self.T:
                    self.update_phi(marginals / t)
                    return
                marginals[player] += self.marginal(S, player, v_S)
                t[player] += 1
            self.update_phi(marginals / t)
        self.func_calls = self.T
        self.save_steps(step_interval)

class CMCS_Dependent(CMCS):
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+2 <= self.T:
            S = self.sample()
            for player in range(n):
                if self.func_calls+2 > self.T:
                    break
                marginals[player] += self.marginal(S, player)
                t[player] += 1
            self.update_phi(marginals / t)
        self.func_calls = self.T
        self.save_steps(step_interval)
        
class CMCS_Independent(CMCS):
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+2 <= self.T:
            for player in range(n):
                if self.func_calls+2 > self.T:
                    break
                S = self.sample()
                marginals[player] += self.marginal(S, player)
                t[player] += 1
            self.update_phi(marginals / t)
            
        self.func_calls = self.T
        self.save_steps(step_interval)
        
class CMCS_Length(CMCS):
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+2 <= self.T:
            length = np.random.randint(n+1)
            for player in range(n):
                if self.func_calls+2 > self.T:
                    break
                S = self.sample(length=length)
                marginals[player] += self.marginal(S, player)
                t[player] += 1
            self.update_phi(marginals / t)
        self.func_calls = self.T
        self.save_steps(step_interval)
        
class Strat_CMCS(CMCS):
    def calc_phi(self, marginals, t):
        count = np.sum(t!=0, axis=1)
        t[t==0] = 1
        return np.sum(marginals / t, axis=1)/count
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros((n, n+1), dtype=np.float32)
        t = np.zeros((n, n+1), dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        
        while self.func_calls+2*(n+1) <= self.T:
            for length in range(n+1):
                S = self.sample(length)
                v_S = self.value(S)
                for player in range(n):
                    if self.func_calls == self.T:
                        self.update_phi(self.calc_phi(marginals, t))
                        return
                    marginals[player, length] += self.marginal(S, player, v_S)    
                    t[player, length] += 1
                self.update_phi(self.calc_phi(marginals, t)) 
            
        self.func_calls = self.T
        self.save_steps(step_interval)

class Variance_CMCS(CMCS):
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        marginals = np.zeros(n, dtype=np.float32)
        
        sum_diff_squared = np.zeros((n,n), dtype=np.float32)
        sum_diff = np.zeros((n,n), dtype=np.float32)
        count_diff = np.zeros((n,n), dtype=np.float32)
        diff_variance = np.zeros((n,n), dtype=np.float32)
        
        selected_players = np.arange(n)
        while self.func_calls+2 <= self.T:
            
            sorted = np.argsort(-self.phi)
            topk, rest = sorted[:k], sorted[k:]
            
            if np.any(count_diff<15) or selected_players.shape[0] == 0:
                selected_players = np.arange(n)
            else:
                sigma_ij = np.sqrt(diff_variance[np.ix_(topk, rest)])
                # diff_ij = self.phi[topk, None] - self.phi[rest]
                diff_ij = (sum_diff/count_diff)[np.ix_(topk, rest)]
                sqrt_mij = np.sqrt(count_diff[np.ix_(topk, rest)])
                cdf_values = norm.cdf(sqrt_mij * (-diff_ij) / sigma_ij, loc=0, scale=1) # the probability that P(phi_i < phi_j), i.e. current estimation is wrong
                
                # pair_idx = np.argmax(cdf_values)
                # selected_players = np.array([topk[pair_idx // (n-k)], rest[pair_idx % (n-k)]])
                
                certainty = 1-cdf_values
                # print(certainty.shape)
                
                min_certainty, max_certainty = np.min(certainty), np.max(certainty)
                if min_certainty == max_certainty: # if all pairs are equally certain just evaluate all
                    selected_players = np.arange(n)
                else:
                    weights = (max_certainty - certainty) / (max_certainty - min_certainty)
                    selected_pairs = (np.random.rand(*certainty.shape) < weights).nonzero()
                    selected_i = np.unique(topk[selected_pairs[0]])
                    selected_j = np.unique(rest[selected_pairs[1]])
                    # print(selected_i, selected_j)
                    selected_players = np.concatenate((selected_i, selected_j))
                
            S = self.sample()
            v_S = self.value(S)
            curr_marginals = np.zeros(selected_players.shape)
            for idx, player in enumerate(selected_players):
                if self.func_calls == self.T:
                    self.update_phi(marginals / t)
                    return
                marginal = self.marginal(S, player, v_S)
                marginals[player] += marginal
                curr_marginals[idx] = marginal
                t[player] += 1
            self.update_phi(marginals / t)
            diffs = curr_marginals[:, None] - curr_marginals
            sum_diff_squared[np.ix_(selected_players, selected_players)] += diffs**2
            sum_diff[np.ix_(selected_players, selected_players)] += diffs
            count_diff[np.ix_(selected_players, selected_players)] += 1
            
            if np.all(count_diff>1):
                diff_variance = (sum_diff_squared - (sum_diff**2)/count_diff)/(count_diff-1)
                # diff_variance = (sum_diff_squared - count_diff*(self.phi[:, None] * self.phi)**2)/(count_diff-1)
                # print(diff_variance)
        self.func_calls = self.T
        self.save_steps(step_interval)