from .algorithm import Algorithm
import numpy as np
import math
from functools import cmp_to_key

class ShapleySort(Algorithm):
    def sample(self):
        length = np.random.randint(self.game.n+1)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length], S[length:]
    def get_value(self, S):
        l = len(S)
        if l == 0:
            v = 0
        elif l == self.game.n:
            v = self.v_n
        else:
            v = self.value(S)
        return v
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        comparisions = np.zeros((n,n))
        t = np.zeros((n,n))
        self.v_n = self.value(np.arange(n))
        def compare(a, b):
            if comparisions[a, b] < comparisions[a, b]:
                return 1
            if comparisions[a, b] == comparisions[a, b]:
                return 0
            return -1
        while self.func_calls+n < self.T:
            S, notS = self.sample()
            values = np.array([self.get_value(S[S != p]) if p in S else self.get_value(np.concatenate((S, [p]))) for p in range(n)])
            
            for p in S:
                for q in S:
                    diff = values[p] - values[q]
                    comparisions[p, q] = (comparisions[p, q] * t[p,q] + diff) / (t[p,q]+1)
                    t[p,q]+=1
            for p in notS:
                for q in notS:
                    diff = values[q] - values[p]
                    comparisions[p, q] = (comparisions[p, q] * t[p,q] + diff) / (t[p,q]+1)
                    t[p,q]+=1
            
            # sorted_players = sorted(np.arange(n), key=cmp_to_key(compare))
            
            pk_pre=-1
            for i in range(n):
                pk = np.argsort(self.phi)[k-1]
                self.phi = comparisions[pk]
                if pk == pk_pre:
                    break
                pk_pre = pk            
            
            self.save_steps(step_interval)
        while self.func_calls < self.T:
            self.func_calls += 1
            self.save_steps(step_interval)