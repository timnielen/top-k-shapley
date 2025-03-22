from .algorithm import PAC_Algorithm
import numpy as np


class SHAP_K(PAC_Algorithm):
    '''
    Implementation of SamplingSHAP@K algorithm.
    Kariyappa, Sanjay, et al. "SHAP@ k: efficient and probably approximately correct (PAC) identification of top-k features." 
    Proceedings of the AAAI Conference on Artificial Intelligence. 
    Vol. 38. No. 12. 2024.
    '''
    def get_top_k(self, k: int):
        n = self.game.n
        
        # warm-up using permutation sampling until t_min is reached and CLT is valid
        for m in range(self.t_min):
            permutation  = np.arange(n)
            np.random.shuffle(permutation)
            pre = self.v_n
            for i in range(n):
                player = permutation[i]
                if(self.func_calls == self.budget):
                    self.save_steps(final=True)
                    return
                value = self.value(permutation[i+1:])
                marginal = pre - value
                self.update_player(player, marginal)
                pre = value
        
        topk, rest = self.partition(k)
        self.update_bounds(topk, rest)
        
        def update_player(player, permutation):
            '''wrapper around self.update_player to update a player using a permutation of [n] instead of a coalition'''
            index = np.argwhere(permutation == player)[0][0]
            marginal = self.value(permutation[index:]) - self.value(permutation[index+1:])
            self.update_player(player, marginal)        
            
        while self.func_calls+4 <= self.budget or (self.budget == -1 and not self.is_PAC()):
            # select the two players of the two partitions with highest overlap in confidence intervals
            h = topk[np.argmin(self.lower_bound[topk])]
            l = rest[np.argmax(self.upper_bound[rest])]
            
            permutation = np.arange(n)
            np.random.shuffle(permutation)
            update_player(h, permutation)
            update_player(l, permutation)
            
            topk, rest = self.partition(k)
            self.update_bounds(topk, rest)
                
        self.save_steps(final=True)