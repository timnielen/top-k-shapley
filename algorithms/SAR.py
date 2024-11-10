from .algorithm import Algorithm
import numpy as np

class SAR(Algorithm):
    def __init__(self, c=1):
        self.c = c
    def get_top_k(self, k: int, step_interval: int = 100):
        n = self.game.n
        self.phi = np.zeros(n)
        t=np.zeros(n)
        topk = []

        def sample(i: int) -> list:
            length = np.random.randint(n)
            S = np.concatenate((np.arange(i), np.arange(i+1, n)))
            np.random.shuffle(S)
            return S[:length]

        num_marginals_pp = np.ceil((self.T - n) / ((n+1-np.arange(n)) * (1/2 + np.sum(1/np.arange(2,n+1)))))
        num_marginals_pp[0] = 0
        round = 1
        available_players = np.arange(n)
        while self.func_calls+len(available_players)*(num_marginals_pp[round] - num_marginals_pp[round-1]) <= self.T and len(topk) < k:
            for player in available_players:
                for _ in range((num_marginals_pp[round] - num_marginals_pp[round-1]).astype(np.int32)):
                    S = sample(player)
                    marginal = self.value(np.concatenate((S, [player]))) - self.value(S)
                    self.phi[player] = (t[player]*self.phi[player] + marginal)/(t[player]+1)
                    t[player] += 1
                    self.save_steps(step_interval)
            
            sorted = np.argsort(-self.phi)
            border = np.zeros(n)
            border[sorted[:k]] = self.phi[sorted[k]]
            border[sorted[k:]] = self.phi[sorted[k-1]]
            delta = np.abs(self.phi - border)
            selected_player = np.argmax(delta) #most certain player
            available_players = available_players[available_players != selected_player]
            if self.phi[player] > self.phi[sorted[k]]:
                topk += [selected_player]
                k -= 1
            round += 1
        # print(self.func_calls, len(topk))
        while self.func_calls < self.T:
            self.func_calls += 1
            self.save_steps(step_interval)