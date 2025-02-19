from .algorithm import PAC_Algorithm
import numpy as np


class SHAP_K(PAC_Algorithm):
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        
        for m in range(self.t_min):
            partition = np.arange(n)
            np.random.shuffle(partition)
            pre = self.v_n
            for i in range(n):
                player = partition[i]
                if i == n-1:
                    value = self.v_0
                else:
                    if(self.func_calls == self.T):
                        return
                    value = self.value(partition[i+1:])
                marginal = pre - value
                self.update_player(player, marginal)
                pre = value
        
        topk, rest = self.partition(k)
        self.update_bounds(topk, rest)
        
        def update_player(player, partition):
            index = np.argwhere(partition == player)[0][0]
            marginal = self.value(partition[index:]) - self.value(partition[index+1:])
            self.update_player(player, marginal)        
            
        while self.func_calls+4 <= self.T or (self.T == -1 and not self.is_PAC()):
            h = topk[np.argmin(self.lower_bound[topk])]
            l = rest[np.argmax(self.upper_bound[rest])]
            
            partition = np.arange(n)
            np.random.shuffle(partition)
            update_player(h, partition)
            update_player(l, partition)
            
            topk, rest = self.partition(k)
            self.update_bounds(topk, rest)
                
        self.save_steps(step_interval, final=True)