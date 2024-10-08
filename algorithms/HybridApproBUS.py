from .algorithm import Algorithm
import numpy as np

class HybridApproBUS(Algorithm):
    def __init__(self, switch: int = 1):
        self.switch = switch
    def get_top_k(self, k: int, step_interval: int = 100):
        n = self.game.n
        self.phi = np.zeros(n)
        t = np.zeros(n)

        def sample() -> list:
            S = np.arange(n)
            np.random.shuffle(S)
            return list(S)

        for r in range(self.switch):
            S = sample()
            pre = 0
            for i in range(n):
                player = S[i]
                if(self.func_calls + 1 > self.T):
                    break 
                t[player] += 1
                value = self.value(S[:i+1])
                phi_new = value - pre
                pre = value
                self.phi[player] = ((t[player]-1)*self.phi[player] + phi_new)/t[player]
                self.save_steps(step_interval)


        while(self.func_calls+2 <= self.T):
            pi = np.argsort(-self.phi)
            border = (self.phi[pi[k-1]] + self.phi[pi[k]])/2
            delta = [abs(self.phi[i] - border) for i in range(n)]
            player = np.argmin([delta[j] * t[j] for j in range(n)])
            t[player] += 1
            S = sample()
            index = np.where(S == player)[0][0]
            phi_new = self.value(S[:index+1]) - self.value(S[:index])
            self.phi[player] = ((t[player]-1)*self.phi[player] + phi_new)/t[player]
            self.save_steps(step_interval)


        if(self.func_calls == self.T-1):
            self.func_calls += 1
        self.save_steps(step_interval)


class SmartHybridApproBUS(Algorithm):
    def __init__(self, switch: int = 1):
        self.switch = switch
    def get_top_k(self, k: int, step_interval: int = 100):
        n = self.game.n
        self.phi = np.zeros(n)
        t = np.zeros(n)

        def sample() -> list:
            S = np.arange(n)
            np.random.shuffle(S)
            return list(S)

        for r in range(self.switch):
            S = sample()
            pre = 0
            for i in range(n):
                player = S[i]
                if(self.func_calls+1 > self.T):
                    break 
                t[player] += 1
                value = self.value(S[:i+1])
                marginal_contribution = value - pre
                pre = value
                new_phi = ((t[player]-1)*self.phi[player] + marginal_contribution)/t[player]
                self.phi[player] = new_phi
                self.save_steps(step_interval)
                

        S = sample()
        sampled_players = [False for _ in range(n)]
        coalitionvalues = [None for _ in range(n)]
        cached_count = 0
        while(self.func_calls <= self.T):
            pi = np.argsort(-self.phi)
            border = (self.phi[pi[k-1]] + self.phi[pi[k]])/2
            delta = [abs(self.phi[i] - border) for i in range(n)]
            player = np.argmin([delta[j] * t[j] for j in range(n)])

            # reset current sample
            if(sampled_players[player] == True):
                S = sample()
                sampled_players = [False for _ in range(n)]
                coalitionvalues = [None for _ in range(n)]
            sampled_players[player] = True

            index = np.where(S == player)[0][0]
            if(index == 0): 
                pre = 0
            else:
                pre = coalitionvalues[S[index-1]]

            if(pre == None):
                if(self.func_calls+1 > self.T):
                    break 
                pre = self.value(S[:index])
                coalitionvalues[S[index-1]] = pre
            else:
                cached_count += 1

            curr = coalitionvalues[player]
            if(curr == None):
                if(self.func_calls+1 > self.T):
                    break 
                curr = self.value(S[:index+1])
                coalitionvalues[player] = curr
            else:
                cached_count += 1

            marginal_contribution = curr - pre
            t[player] += 1


            new_phi = ((t[player]-1)*self.phi[player] + marginal_contribution)/t[player]
            self.phi[player] = new_phi
            self.save_steps(step_interval)

        if(self.func_calls == self.T-1):
            self.func_calls += 1
        self.save_steps(step_interval)