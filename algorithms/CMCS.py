from .algorithm import Algorithm
import math
import numpy as np

class CMCS(Algorithm):
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
        
def binom(n, k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
class SIR_CMCS(Algorithm):
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
        
        player_weights = np.zeros(n)
        while self.func_calls+n+1 <= self.T/2:
            S, notS = self.sample()
            v_S = self.value(S)
            for player in S:
                marginal = v_S - self.value(S[S != player])
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
                player_weights[S] += abs(marginal)
            for player in notS:
                marginal = self.value(np.concatenate((S, [player]))) - v_S
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
                player_weights[notS] += abs(marginal)
            t += 1
        # for p in range(n):
        #     assert player_weights[p] > 0
        pre_samples = 100
        total_weights = player_weights.mean()
        for player in range(n):
            for length in range(1, n+1):
                total_weights += binom(n-1, length-1) * player_weights[player] / length
        # player_weights /= total_weights
        coalition_weights = np.array([1/((n+1)*binom(n, l)) for l in range(n+1)])
        
        self.phi = np.zeros(n)
        t = 0
        while self.func_calls+n+1 <= self.T:
            total_weight_pre = 0
            for i in range(pre_samples):
                S_new, notS_new = self.sample()
                length_new = S_new.shape[0]
                if length_new == 0:
                    weight = player_weights.mean()
                else:
                    weight = player_weights[S_new].mean()
                # weight /= coalition_weights[length]
                total_weight_pre += weight
                if np.random.rand() < weight/total_weight_pre:
                    S, notS = S_new, notS_new
                    length = length_new
            if length == 0:
                pdf = player_weights.mean() / total_weights
            else:
                pdf = player_weights[S].mean() / total_weights
            v_S = self.value(S)
            for player in S:
                marginal = v_S - self.value(S[S != player])
                marginal *= 1 / pdf
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
            for player in notS:
                marginal = self.value(np.concatenate((S, [player]))) - v_S
                marginal *= 1 / pdf
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
            t += 1
            
        self.func_calls = self.T
        self.save_steps(step_interval)