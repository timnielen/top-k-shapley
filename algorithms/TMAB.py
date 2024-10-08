from .algorithm import Algorithm
import numpy as np

class TMAB(Algorithm):
    def __init__(self, trunc_border = 1e-5, delta = 0.2, epsilon=1e-4):
        self.trunc_border = trunc_border
        self.trunc_l = 0
        self.delta = delta
        self.epsilon=epsilon
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        t = np.zeros(n)
        def sample():
            S = np.arange(n)
            np.random.shuffle(S)
            return S
        chosen_players = np.arange(n)
        variances = np.ones(n)
        sums = np.zeros(n)
        squares = np.zeros(n)
        cbs = np.ones(n)
        while self.func_calls < self.T and len(chosen_players) > 0:
            S = sample()
            pre = None
            for j in range(n):
                p = S[j]
                if p in chosen_players:
                    if pre == None:
                        if self.func_calls == self.T:
                            self.save_steps(step_interval)
                            #print(chosen_players.shape)     
                            return
                        pre = self.value(S[j:])
                    if pre < self.trunc_border:
                        print("truncate")
                        break
                    if self.func_calls == self.T:
                        self.save_steps(step_interval)
                        #print(chosen_players.shape)
                        return
                    v = self.value(S[j+1:])
                    #assert v >= 0 and v <= 1, v
                    marginal = pre - v
                    t[p] += 1
                    sums[p] += marginal
                    squares[p] += marginal**2
                    self.phi[p] = sums[p]/t[p]
                    pre = v
                    self.save_steps(step_interval)
                else:
                    pre = None
            t = np.clip(t, 1e-12, None)
            variances = np.ones(n)
            variances[t > 1] = squares[t > 1]
            variances[t > 1] -= (sums[t > 1] ** 2)/t[t > 1]
            variances[t > 1] /= (t[t > 1] - 1)
            cbs = np.ones(n)
            cbs[t > 1] = np.sqrt(2 * variances[t > 1] * np.log(2 / self.delta) / t[t > 1]) +\
            7/3 * np.log(2 / self.delta) / (t[t > 1] - 1)
            thresh = (self.phi)[np.argsort(self.phi)[-k-1]]
            chosen_players = np.where(
                ((self.phi - cbs + self.epsilon) < thresh) * ((self.phi + cbs - self.epsilon) > thresh))[0]
        print(self.delta, chosen_players.shape)
        while self.func_calls < self.T:
            self.func_calls += 1
            self.save_steps(step_interval)
