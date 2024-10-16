from .algorithm import Algorithm
import numpy as np

class SCM(Algorithm):
    def value(self, S: list):
        l = len(S)
        if l == 0:
            v = self.v_0
        elif l == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.T
            self.func_calls += 1
            v = self.game.value(S)
        return v
    def sample(self):
        length = np.random.choice(np.arange(self.game.n+1))
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length], S[length:]
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        t = 0
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+n+1 <= self.T:
            S, notS = self.sample()
            v_S = self.value(S)
            for player in S:
                marginal = v_S - self.value(S[S != player])
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
            for player in notS:
                marginal = self.value(np.concatenate((S, [player]))) - v_S
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
            t += 1
        self.func_calls = self.T
        self.save_steps(step_interval)