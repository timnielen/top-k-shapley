from .algorithm import Algorithm
import numpy as np

class GapE(Algorithm):
    def __init__(self, c=1):
        self.c = c
    def get_top_k(self, k: int, step_interval: int = 100):
        n = self.game.n
        self.phi = np.zeros(n)
        actual_phi = np.array([self.game.get_phi(i) for i in range(n)])
        sorted = np.argsort(-actual_phi)
        border = np.zeros(n)
        border[sorted[:k]] = actual_phi[sorted[k]]
        border[sorted[k:]] = actual_phi[sorted[k-1]]
        delta = np.abs(actual_phi - border)
        # print(actual_delta[actual_sorted]**2)
        heuristic = np.sum(1/(delta**2))
        # print(actual_phi[actual_sorted])
        # print(actual_delta)
        # print(actual_delta**2)
        # print(1/(actual_delta**2))
        # print()
        t=np.zeros(n)

        def sample(i: int) -> list:
            length = np.random.randint(n)
            S = np.concatenate((np.arange(i), np.arange(i+1, n)))
            np.random.shuffle(S)
            return S[:length]
        
        for player in range(n):
            if(self.func_calls+2 > self.T):
                break
            S = sample(player)
            t[player] += 1
            self.phi[player] = self.value(np.concatenate((S, [player]))) - self.value(S)
            self.save_steps(step_interval)

        while(self.func_calls+2 <= self.T):
            sorted = np.argsort(-self.phi)
            border = np.zeros(n)
            border[sorted[:k]] = self.phi[sorted[k]]
            border[sorted[k:]] = self.phi[sorted[k-1]]
            delta = np.abs(self.phi - border)
            selected_player = np.argmax(-delta + self.c*np.sqrt((self.T / heuristic) / t))  
            # selected_player = np.argmin(t * delta)#* heuristic)))
            S = sample(selected_player)
            marginal = self.value(np.concatenate((S, [selected_player]))) - self.value(S)
            self.phi[selected_player] = (t[selected_player]*self.phi[selected_player] + marginal)/(t[selected_player]+1)
            t[selected_player] += 1
            self.save_steps(step_interval)