from .algorithm import Algorithm
import numpy as np

class ApproShapley(Algorithm):
    def __init__(self, optimize: bool=True):
        self.optimize = optimize
    def get_top_k(self, k: int, step_interval: int = 100):
        if(self.optimize):
            return self.optimized(k, step_interval)
        return self.not_optimized(k, step_interval)
    def optimized(self, k: int, step_interval: int = 100):
        n = self.game.n
        self.phi = np.zeros(n)
        t=0
        v_n = self.value(np.arange(n))
        v_0 = self.value(np.array([]))
        while(True):
            O = np.arange(n)
            np.random.shuffle(O)
            pre = v_n
            for i in range(n):
                player = O[i]
                if i == n-1:
                    value = v_0
                else:
                    if(self.func_calls == self.T):
                        return
                    value = self.value(O[i+1:])
                self.phi[player] = (t*self.phi[player] + pre - value)/(t+1)
                pre = value
                self.save_steps(step_interval)
            t += 1
            
    
    def not_optimized(self, k: int, step_interval: int = 100):
        n = self.game.n
        self.phi = np.zeros(n)
        t=np.zeros(n)
        while(self.func_calls+2 <= self.T):
            O = np.arange(n)
            np.random.shuffle(O)
            for i in range(n):
                if(self.func_calls+2 > self.T):
                    break
                player = O[i]
                t[player] += 1
                m = self.value(O[:i+1]) - self.value(O[:i])
                self.phi[player] = ((t[player]-1)*self.phi[player] + m)/t[player]
                self.save_steps(step_interval)