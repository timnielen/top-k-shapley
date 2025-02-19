from .algorithm import Algorithm, PAC_Algorithm
import math
import numpy as np
import util
import scipy.stats

class CMCS(PAC_Algorithm):
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
        
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        
        while self.func_calls+2 <= self.T or (self.T == -1 and not self.is_PAC()):
            S = self.sample()
            v_S = self.value(S)
            for player in range(n):
                if self.func_calls == self.T:
                    return
                marginal = self.marginal(S, player, v_S)
                self.update_player(player, marginal)
            
            topk, rest = self.partition(k)
            self.update_bounds(topk, rest)
        
        self.save_steps(step_interval, final=True)

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
        self.save_steps(step_interval, final=True)

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
        self.save_steps(step_interval, final=True)
        
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
            
        self.save_steps(step_interval, final=True)
        
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
        
        self.save_steps(step_interval, final=True)
        
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
            
        
        self.save_steps(step_interval, final=True)

class Greedy_CMCS(CMCS):
    def select_players(self, sum_diff_squared, sum_diff, count_diff, topk, rest):
        n = self.game.n
        selected_players = np.arange(n)
        if np.any(count_diff < self.t_min):
            return selected_players
        
        diff_variance = (sum_diff_squared - (sum_diff**2)/count_diff)/(count_diff-1)
        cdf_values = scipy.stats.norm.cdf(-sum_diff / np.sqrt(diff_variance*count_diff), loc=0, scale=1)
        
        min_cdf, max_cdf = np.min(cdf_values), np.max(cdf_values)
        if min_cdf != max_cdf: # be greedy only if pairs have different probabilities 
            weights = (cdf_values - min_cdf) / (max_cdf - min_cdf)
            selected_pairs = (np.random.rand(*cdf_values.shape) < weights).nonzero()
            # get unique indices and convert indices of topk and rest sets to indices of full set
            selected_i = np.unique(topk[selected_pairs[0]]) 
            selected_j = np.unique(rest[selected_pairs[1]])
            selected_players = np.concatenate((selected_i, selected_j))
            
        return selected_players
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        
        sum_diff_squared = np.zeros((n,n), dtype=np.float32)
        sum_diff = np.zeros((n,n), dtype=np.float32)
        count_diff = np.zeros((n,n), dtype=np.float32)
        
        topk, rest = self.partition(k)
        
        while self.func_calls+2 <= self.T or (self.T == -1 and not self.is_PAC()):
            
            indices = np.ix_(topk, rest) # only select from i in topk and j in rest
            selected_players = self.select_players(sum_diff_squared[indices], sum_diff[indices], count_diff[indices], topk, rest)
                            
            S = self.sample()
            v_S = self.value(S)
            
            curr_marginals = np.zeros(selected_players.shape)
            for idx, player in enumerate(selected_players):
                if self.func_calls == self.T:
                    return
                marginal = self.marginal(S, player, v_S)
                self.update_player(player, marginal)
                curr_marginals[idx] = marginal
            
            indices = np.ix_(selected_players, selected_players) # select i,j in selected players
            diffs = curr_marginals[:, None] - curr_marginals
            sum_diff_squared[indices] += diffs**2
            sum_diff[indices] += diffs
            count_diff[indices] += 1
            
            topk, rest = self.partition(k)
            self.update_bounds(topk, rest)
            
        self.save_steps(step_interval, final=True)
           
        
class CMCS_at_K(CMCS):
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
            
        for m in range(self.t_min):
            S = self.sample()
            v_S = self.value(S)
            for player in range(n):
                marginal = self.marginal(S, player, v_S)
                self.update_player(player, marginal)
        
        topk, rest = self.partition(k)
        self.update_bounds(topk, rest)
        while self.func_calls+4 <= self.T or (self.T == -1 and not self.is_PAC()):
            
            h = topk[np.argmin(self.lower_bound[topk])]
            l = rest[np.argmax(self.upper_bound[rest])]
            
            S = self.sample()
            v_S = self.value(S)
            
            marginal = self.marginal(S, h, v_S)
            self.update_player(h, marginal)
            
            marginal = self.marginal(S, l, v_S)
            self.update_player(l, marginal)
            
            topk, rest = self.partition(k)
            self.update_bounds(topk, rest)
                
        self.save_steps(step_interval, final=True)
        
