from .algorithm import Algorithm
import numpy as np
import math

class SVARM(Algorithm):
    '''
    implementation of SVARM
    Kolpaczki, Patrick, et al. "Approximating the shapley value without marginal contributions." 
    Proceedings of the AAAI conference on Artificial Intelligence. 
    Vol. 38. No. 12. 2024.
    '''
    def get_top_k(self, k: int):
        assert self.budget != -1, "The algorithm doesn't have an early stopping condition!"
        n = self.game.n

        # inititalize estimators for positive and negative partial marginals
        phi_plus = np.zeros(n, dtype=np.float32)
        phi_minus = np.zeros(n, dtype=np.float32)
        c_plus = np.ones(n, dtype=np.int32)
        c_minus = np.ones(n, dtype=np.int32)
        
        H_n = sum([1/k for k in range(1,n+1)])
        def P_plus():
            length = np.random.choice(np.arange(1,n+1), p=[1/(l*H_n) for l in range(1,n+1)])
            S = np.arange(n)
            np.random.shuffle(S)
            return S[:length]
        def P_minus():
            length = np.random.choice(np.arange(n), p=[1/((n-l)*H_n) for l in range(n)])
            S = np.arange(n)
            np.random.shuffle(S)
            return S[:length]
        #WarmUp
        for player in range(n):
            A_plus = self.sample(player)
            A_minus = self.sample(player)
            phi_plus[player] = self.value(np.concatenate((A_plus, [player])))
            phi_minus[player] = self.value(A_minus)
            self.phi = phi_plus - phi_minus
            self.save_steps()
        
        while self.func_calls+2 <= self.budget:
            A_plus = P_plus()
            A_minus = P_minus()
            v_plus = self.value(A_plus)
            v_minus = self.value(A_minus)
            for i in A_plus:
                phi_plus[i] = (c_plus[i]*phi_plus[i]+v_plus)/(c_plus[i]+1)
                c_plus[i] += 1
            for i in [i for i in range(n) if not i in A_minus]:
                phi_minus[i] = (c_minus[i]*phi_minus[i]+v_minus)/(c_minus[i]+1)
                c_minus[i] += 1

            self.phi = phi_plus - phi_minus
            self.save_steps()
        self.save_steps(final = True)

class StratSVARM(Algorithm):
    '''
    implementation of Stratified SVARM
    Kolpaczki, Patrick, et al. "Approximating the shapley value without marginal contributions." 
    Proceedings of the AAAI conference on Artificial Intelligence. 
    Vol. 38. No. 12. 2024.
    '''
    def __init__(self, start_exact=False, theoretical_distribution=False):
        self.start_exact = start_exact
        self.theoretical_distribution = theoretical_distribution

    def p_size(self, s):
        '''theoretical distribution of coalition sizes defined by the authors'''
        n = self.game.n
        if n%2 == 0:
            H = sum([1/k for k in range(1,math.floor(n/2))])
            if s in [i for i in range(2,math.floor((n-2)/2+1))]:
                return (n*math.log(n)-1)/(2*s*n*math.log(n)*(H-1))
            if s == n/2:
                return 1/(n*math.log(n))
            else:
                return (n*math.log(n)-1)/(2*(n-s)*n*math.log(n)*(H-1))
        else:
            H = sum([1/k for k in range(1,math.floor((n-1)/2+1))])
            if s in [i for i in range(2,math.floor((n-1)/2+1))]:
                return 1/(2*s*(H-1))
            else: 
                return 1/(2*(n-s)*(H-1))

    def update(self, S, notS):
        '''Updates all players using the value of a coalition S. Updates positive estimator if player in S else negative estimator'''
        l = len(S)
        v = self.value(S)
        for player in S:
            self.phi_p[player, l-1] = (self.count_p[player, l-1] * self.phi_p[player, l-1] + v) / (self.count_p[player, l-1] + 1)
            self.count_p[player, l-1] += 1
        for player in notS:
            self.phi_m[player, l] = (self.count_m[player, l] * self.phi_m[player, l] + v) / (self.count_m[player, l] + 1)
            self.count_m[player, l] += 1
            
        self.phi = np.sum(self.phi_p-self.phi_m,axis=1)/self.game.n
        self.save_steps()
    def sample(self):
        n = self.game.n
        if not self.start_exact:
            length = np.random.randint(1, n)
        elif self.theoretical_distribution:
            length = np.random.choice(np.arange(2,n-1), p=self.distribution)
        else: 
            length = np.random.randint(2, n-1)
        S = np.arange(n)
        np.random.shuffle(S)
        return S[:length], S[length:]
    def get_top_k(self, k: int):
        n = self.game.n
        self.phi = np.zeros(n)
        self.phi_p = np.zeros((n,n))
        self.phi_m = np.zeros((n,n))
        self.count_p = np.zeros((n,n))
        self.count_m = np.zeros((n,n))
        self.update(np.arange(n), np.array([]))
        self.update(np.array([]), np.arange(n))
        self.distribution = np.array([self.p_size(s) for s in range(2,n-1)])
        if self.start_exact:
            # evaluate stratas of length 1 and n-1 exactly
            for player in range(n):
                not_player = np.concatenate((np.arange(player), np.arange(player+1, n)))
                self.update(not_player, np.array([player]))
                self.update(np.array([player]), not_player)
                
        while self.func_calls + 2 <= self.budget:
            S, notS = self.sample()
            self.update(S, notS)
            self.update(notS, S)
            
        self.save_steps(final=True)
