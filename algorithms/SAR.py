from .algorithm import Algorithm
import numpy as np

class SAR(Algorithm):
    '''
    Successive Accepts and Rejects (SAR) modified for the Topk problem.
    Bubeck, S., Wang, T., Viswanathan, N.: Multiple identifications in multi-armed
    bandits. In: Proceedings of the 30th International Conference on Machine Learning
    (ICML). pp. 258–265 (2013)
    '''
    def __init__(self, c=1):
        self.c = c
    def get_top_k(self, k: int):
        assert self.budget != -1, "The algorithm doesn't have an early stopping condition!"
        n = self.game.n
        m = k
        topk = []

        num_marginals_pp = np.ceil((self.budget - n) / ((n+1-np.arange(n)) * (1/2 + np.sum(1/np.arange(2,n+1)))))
        num_marginals_pp[0] = 0
        round = 1
        available_players = np.arange(n)
        while self.func_calls+2 <= self.budget and len(topk) < k:
            for player in available_players:
                for _ in range((num_marginals_pp[round] - num_marginals_pp[round-1]).astype(np.int32)):
                    if self.func_calls + 2 > self.budget:
                        self.save_steps(final=True)
                        return
                    S = self.sample(player)
                    marginal = self.value(np.concatenate((S, [player]))) - self.value(S)
                    self.update_player(player, marginal)
            
            phi = self.phi[available_players]
            sorted = np.argsort(-phi)
            border = np.zeros(len(available_players))
            border[sorted[:m]] = phi[sorted[m]]
            border[sorted[m:]] = phi[sorted[m-1]]
            delta = np.abs(phi - border)
            selected_player = available_players[np.argmax(delta)] # most certain player
            available_players = available_players[available_players != selected_player]
            if self.phi[selected_player] > phi[sorted[m]]:
                topk += [selected_player]
                m -= 1
            round += 1
            
        self.save_steps(final=True)