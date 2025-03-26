from .algorithm import Algorithm
import numpy as np

class compSHAP(Algorithm):
    def sample(self):
        length = np.random.randint(1, self.game.n)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length], S[length:]
    
    # def update(self, coalition):
    #     if value_coalition is None:
    #         value_coalition = self.value(coalition)
    #     if player in coalition:
    #         return value_coalition - self.value(coalition[coalition != player])
    #     return self.value(np.concatenate((coalition, [player]))) - value_coalition
        
    def get_top_k(self, k: int):
        n = self.game.n
        diff = np.zeros((n,n), dtype=np.float32)
        squared_diff = np.zeros((n,n), dtype=np.float32)
        count = 2*np.eye(n, dtype=np.uint32)

        while self.func_calls < self.budget:
            S, notS = self.sample()

            v1 = self.value(S)
            idx1 = np.ix_(S, notS)
            idx2 = np.ix_(notS, S)
            diff[idx1] -= v1
            diff[idx2] += v1
            squared_diff[idx1] += v1*v1
            squared_diff[idx2] += v1*v1
            count[idx1] += 1
            count[idx2] += 1
            phi_approx = diff / count

            # find the player that is most confident in its ordering
            idx = np.all(count > 1, axis=1)
            if np.all(idx == False):
                border_player = np.random.randint(n)
            else:
                sigma = (squared_diff[idx]-count[idx]*(phi_approx[idx]**2))/(count[idx]-1)
                sum_sigma = np.sum(sigma, axis=1)
                border_player = np.arange(n)[idx][np.argmin(sum_sigma)]

            self.phi = phi_approx[border_player]
            self.save_steps()