from .algorithm import Algorithm, PAC_Algorithm
import math
import numpy as np
import util
import scipy.stats

class CMCS(PAC_Algorithm):
    def sample(self, length=None):
        ''' 
        - samples a coalition according to CMCS distribution
        - if a length is defined samples a random coalition using this length (only valid for length ablation)
        '''
        if length is None:
            length = np.random.randint(self.game.n+1)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length]
    
    def marginal(self, coalition, player, value_coalition=None):
        '''
        evaluates the extended marginal contribution of a player (using a precomputed coalition value if defined)
        '''
        if value_coalition is None:
            value_coalition = self.value(coalition)
        if player in coalition:
            return value_coalition - self.value(coalition[coalition != player])
        return self.value(np.concatenate((coalition, [player]))) - value_coalition
        
    def get_top_k(self, k: int):
        n = self.game.n
        
        while self.func_calls+2 <= self.budget or (self.budget == -1 and not self.is_PAC()):
            S = self.sample()
            v_S = self.value(S) # cache coalition value to be used for each player
            for player in range(n):
                if self.func_calls == self.budget:
                    return
                marginal = self.marginal(S, player, v_S)
                self.update_player(player, marginal)
            
            topk, rest = self.partition(k)
            self.update_bounds(topk, rest)
        
        self.save_steps(final=True)

class CMCS_Dependent(CMCS):
    '''
    CMCS ablation where no coalition values are reused
    '''

    def get_top_k(self, k: int):
        n = self.game.n
        
        while self.func_calls+2 <= self.budget or (self.budget == -1 and not self.is_PAC()):
            S = self.sample()
            for player in range(n):
                if self.func_calls+2 > self.budget:
                    break
                marginal = self.marginal(S, player)
                self.update_player(player, marginal)
                
            topk, rest = self.partition(k)
            self.update_bounds(topk, rest)

        self.save_steps(final=True)
        
        
class CMCS_Independent(CMCS):
    '''
    CMCS ablation where coalitions are sampled independently and no coalition values are reused. (I.e. just a naive shapley value sampling algorithm)
    '''

    def get_top_k(self, k: int):
        n = self.game.n
        
        while self.func_calls+2 <= self.budget or (self.budget == -1 and not self.is_PAC()):
            for player in range(n):
                if self.func_calls+2 > self.budget:
                    break
                S = self.sample()
                marginal = self.marginal(S, player)
                self.update_player(player, marginal)

            topk, rest = self.partition(k)
            self.update_bounds(topk, rest)
            
        self.save_steps(final=True)
        
class CMCS_Length(CMCS):
    '''
    CMCS ablation where random coalitions of same length are sampled for each player and no coalition values are reused.
    '''

    def get_top_k(self, k: int):
        n = self.game.n
        
        while self.func_calls+2 <= self.budget or (self.budget == -1 and not self.is_PAC()):
            length = np.random.randint(n+1)
            for player in range(n):
                if self.func_calls+2 > self.budget:
                    break
                S = self.sample(length=length)
                marginal = self.marginal(S, player)
                self.update_player(player, marginal)

            topk, rest = self.partition(k)
            self.update_bounds(topk, rest)
        
        self.save_steps(final=True)

class Greedy_CMCS(CMCS):
    '''
    CMCS version that greedily selects players based on the probability of being partitioned incorrectly
    '''

    def select_players(self, sum_diff_squared, sum_diff, count_diff, topk, rest):
        '''
        select players based on the probability of pairs (i,j) being partitioned incorrectly, given i in current topk and j in non-topk
        '''
        n = self.game.n
        selected_players = np.arange(n)
        if np.any(count_diff < self.t_min): # as long as t_min is not reached (i.e. the CLT is not yet valid) update/select all players
            return selected_players
        
        diff_variance = (sum_diff_squared - (sum_diff**2)/count_diff)/(count_diff-1) # calculate sample variance of the pairwise distances phi_i - phi_j
        cdf_values = scipy.stats.norm.cdf(-sum_diff / np.sqrt(diff_variance*count_diff), loc=0, scale=1) # use CLT to approximate P(phi_i < phi_j) given i in topk and j in non-topk
        cdf_values[diff_variance == 0] = 0 # if we have zero variance current partition is probably correct
        
        min_cdf, max_cdf = np.min(cdf_values), np.max(cdf_values)
        if min_cdf != max_cdf: # be greedy only if pairs have different probabilities 
            weights = (cdf_values - min_cdf) / (max_cdf - min_cdf)
            selected_pairs = (np.random.rand(*cdf_values.shape) < weights).nonzero() # select pairs based on heuristic distribution
            # convert indices of topk and non-topk sets to indices of full set and remove duplicates
            selected_i = np.unique(topk[selected_pairs[0]]) 
            selected_j = np.unique(rest[selected_pairs[1]])
            selected_players = np.concatenate((selected_i, selected_j))
            
        return selected_players
    
    def get_top_k(self, k: int):
        n = self.game.n
        
        sum_diff_squared = np.zeros((n,n), dtype=np.float32)
        sum_diff = np.zeros((n,n), dtype=np.float32)
        count_diff = np.zeros((n,n), dtype=np.float32)
        
        topk, rest = self.partition(k)
        
        while self.func_calls+2 <= self.budget or (self.budget == -1 and not self.is_PAC()):
            
            indices = np.ix_(topk, rest) # only select from i in topk and j in non-topk
            selected_players = self.select_players(sum_diff_squared[indices], sum_diff[indices], count_diff[indices], topk, rest)
                            
            S = self.sample()
            v_S = self.value(S)
            
            curr_marginals = np.zeros(selected_players.shape)
            for idx, player in enumerate(selected_players):
                if self.func_calls == self.budget:
                    return
                marginal = self.marginal(S, player, v_S)
                self.update_player(player, marginal)
                curr_marginals[idx] = marginal
            
            # update all pairs (i,j) in (selected players x selected players)
            indices = np.ix_(selected_players, selected_players) 
            diffs = curr_marginals[:, None] - curr_marginals
            sum_diff_squared[indices] += diffs**2
            sum_diff[indices] += diffs
            count_diff[indices] += 1
            
            topk, rest = self.partition(k)
            self.update_bounds(topk, rest)
            
        self.save_steps(final=True)
           
        
class CMCS_at_K(CMCS):
    '''
    - CMCS version that greedily selects two players, one from each partition, based on their overlap confidence intervals 
    - base algorithm from SHAP@K paper using CMCS as sampling algorithm
    '''
    def get_top_k(self, k: int):
        n = self.game.n
        
        # warm-up by updating all players until CLT is valid, i.e. t_min is reached
        for m in range(self.t_min):
            S = self.sample()
            v_S = self.value(S)
            for player in range(n):
                if self.func_calls == self.budget:
                    return
                marginal = self.marginal(S, player, v_S)
                self.update_player(player, marginal)
        
        topk, rest = self.partition(k)
        self.update_bounds(topk, rest)
        while self.func_calls+4 <= self.budget or (self.budget == -1 and not self.is_PAC()):
            
            # select the two players of the two partitions with highest overlap in confidence intervals
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
                
        self.save_steps(final=True)
        
