from .algorithm import Algorithm
import numpy as np

class ApproShapley(Algorithm):
    def __init__(self, optimize: bool=True):
        self.optimize = optimize
    def get_top_k(self, k: int):
        assert self.budget != -1, "The algorithm doesn't have an early stopping condition!"
        if(self.optimize): # the optimized version reuses coalition values along the permutation
            return self.optimized(k)
        return self.not_optimized(k)
    
    def optimized(self, k: int):
        n = self.game.n
        while(True):
            permutation = np.arange(n)
            np.random.shuffle(permutation)
            pre = self.v_n
            for i in range(n):
                player = permutation[i]
                if(self.func_calls == self.budget):
                    self.save_steps(final = True)
                    return
                value = self.value(permutation[i+1:])
                marginal = pre - value
                self.update_player(player, marginal)
                pre = value
    
    def not_optimized(self, k: int):
        n = self.game.n
        while(True):
            permutation = np.arange(n)
            np.random.shuffle(permutation)
            for i in range(n):
                if(self.func_calls+2 > self.budget):
                    self.save_steps(final = True)
                    return
                player = permutation[i]
                marginal = self.value(permutation[:i+1]) - self.value(permutation[:i])
                self.update_player(player, marginal)