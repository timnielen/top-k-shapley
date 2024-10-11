from .algorithm import Algorithm
import numpy as np

class BUS(Algorithm):
    def get_top_k(self, k: int, step_interval: int = 100):
        n = self.game.n
        self.phi = np.zeros(n)
        t=np.zeros(n)

        # def sample(i: int) -> list:
        #     S = np.arange(n)
        #     np.random.shuffle(S)
        #     index = np.where(S == i)[0][0]
        #     return list(S[:index])
        def sample(i: int) -> list:
            length = np.random.choice(np.arange(n))
            S = np.concatenate((np.arange(i), np.arange(i+1, n)))
            np.random.shuffle(S)
            return S[:length]
        
        for i in range(n):
            if(self.func_calls+2 > self.T):
                break
            S = sample(i)
            t[i] += 1
            self.phi[i] = self.value(np.concatenate((S, [i]))) - self.value(S)
            self.save_steps(step_interval)


        while(self.func_calls+2 <= self.T):
            pi = np.argsort(-self.phi)
            border = (self.phi[pi[k-1]] + self.phi[pi[k]])/2
            delta = [abs(self.phi[i] - border) for i in range(n)]
            i = np.argmin([delta[j] * t[j] for j in range(n)])
            t[i] += 1
            S = sample(i)
            phi_new = self.value(np.concatenate((S, [i]))) - self.value(S)
            self.phi[i] = ((t[i]-1)*self.phi[i] + phi_new)/t[i]
            self.save_steps(step_interval)