from .algorithm import Algorithm
import numpy as np

class BUS(Algorithm):
    def get_top_k(self, k: int):
        assert self.budget != -1, "The algorithm doesn't have an early stopping condition!"
        n = self.game.n
        
        # warm-up round: update each player once
        for player in range(n):
            if(self.func_calls+2 > self.budget):
                break
            S = self.sample(player)
            marginal = self.value(np.concatenate((S, [player]))) - self.value(S)
            self.update_player(player, marginal)

        while(self.func_calls+2 <= self.budget):
            sorted = np.argsort(-self.phi)
            border = (self.phi[sorted[k-1]] + self.phi[sorted[k]])/2
            dist = [abs(self.phi[i] - border) for i in range(n)] 

            player = np.argmin([dist[j] * self.t[j] for j in range(n)]) # select player based on uncertainty
            S = self.sample(player)
            marginal = self.value(np.concatenate((S, [player]))) - self.value(S)
            self.update_player(player, marginal)