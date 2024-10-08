from .algorithm import Algorithm
import numpy as np
from functools import cmp_to_key

class TopkRM(Algorithm):
    def save_steps(self, step_interval: int):
        if(self.func_calls/step_interval >= len(self.values) + 1):
            self.sorted = sorted(self.sorted, key=cmp_to_key(self.compare))
            self.phi[self.sorted] = np.arange(self.game.n)
            self.values += [np.array(self.phi)]
    def compare(self, i1, i2):
            if self.comp[i1, i2] < self.comp[i2, i1]:
                return -1
            if self.comp[i1, i2] == self.comp[i2, i1]:
                return 0
            return 1
    def get_top_k(self, k: int, step_interval: int = 100):
        n = self.game.n
        self.phi = np.zeros(n)
        self.sorted = list(np.arange(n))
        self.comp = np.zeros((n,n))
        self.count = np.zeros((n,n))

        while self.func_calls <self.T:
            length = np.random.choice(np.arange(1,n))
            S = np.arange(n)
            np.random.shuffle(S)
            S = S[:length]
            val = self.value(S)
            notS = np.array([i for i in range(n) if not i in S])
            
            for i in S:
                for j in notS:
                    self.comp[i,j] = (self.count[i,j] * self.comp[i,j] + val)/(self.count[i,j] + 1)
                    self.count[i,j] += 1
            self.save_steps(step_interval)