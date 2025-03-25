from .algorithm import Algorithm
import numpy as np

class GapE(Algorithm):
    def __init__(self, c=1):
        self.c = c
    def get_top_k(self, k: int):
        assert self.budget != -1, "The algorithm doesn't have an early stopping condition!"
        n = self.game.n
        actual_phi = self.game.get_phi()
        sorted = np.argsort(-actual_phi)
        border = np.zeros(n)
        border[sorted[:k]] = actual_phi[sorted[k]]
        border[sorted[k:]] = actual_phi[sorted[k-1]]
        delta = np.abs(actual_phi - border)
        heuristic = np.sum(1/(delta**2))
        
        # warm-up round: update each player once
        for player in range(n):
            if(self.func_calls+2 > self.budget):
                break
            S = self.sample(player)
            marginal = self.value(np.concatenate((S, [player]))) - self.value(S)
            self.update_player(player, marginal)

        while(self.func_calls+2 <= self.budget):
            sorted = np.argsort(-self.phi)
            border = np.zeros(n)
            border[sorted[:k]] = self.phi[sorted[k]]
            border[sorted[k:]] = self.phi[sorted[k-1]]
            delta = np.abs(self.phi - border)
            selected_player = np.argmax(-delta + self.c*np.sqrt((self.budget / heuristic) / self.t))  
            S = self.sample(selected_player)
            marginal = self.value(np.concatenate((S, [selected_player]))) - self.value(S)
            self.update_player(selected_player, marginal)

        self.save_steps(final=True)