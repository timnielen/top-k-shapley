from .algorithm import Algorithm
import numpy as np
import math

class SVARM(Algorithm):
    def get_top_k(self, k: int, step_interval: int = 100):
        n = self.game.n
        self.phi = np.zeros(n)
        
        phi_plus = np.zeros(n)
        phi_minus = np.zeros(n)

        c_plus = np.ones(n)
        c_minus = np.ones(n)
        
        
        def P_w(i: int):
            length = np.random.choice(np.arange(n))
            S = np.concatenate((np.arange(i), np.arange(i+1,n)))
            np.random.shuffle(S)
            return S[:length]
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
        for i in range(n):
            A_plus = P_w(i)
            A_minus = P_w(i)
            phi_plus[i] = self.value(np.concatenate((A_plus, [i])))
            phi_minus[i] = self.value(A_minus)
            self.phi = np.array(phi_plus) - np.array(phi_minus)
            self.save_steps(step_interval)
        
        while self.func_calls+2 <= self.T:
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

            self.phi = np.array(phi_plus) - np.array(phi_minus)
            self.save_steps(step_interval)

class oldStratSVARM(Algorithm):
    def __init__(self, start_exact=True):
        self.start_exact = start_exact
    def update(self, A: list, A_p: list, A_m: list):
        v = self.value(A)
        for i in [i for i in A_p if i in A]:
            self.phi_p[i, len(A)-1] = (self.c_p[i, len(A)-1] * self.phi_p[i, len(A)-1] + v)/(self.c_p[i, len(A)-1]+1)
            self.c_p[i, len(A)-1] += 1

        for i in [i for i in A_m if not i in A_p]:
            self.phi_m[i, len(A)] = (self.c_m[i, len(A)] * self.phi_m[i, len(A)] + v)/(self.c_m[i, len(A)]+1)
            self.c_m[i, len(A)] += 1
        
        self.phi = 1/self.game.n * np.sum(self.phi_p-self.phi_m,axis=1)
        self.save_steps(self.step_interval)

    def exact_calculation(self):
        n = self.game.n
        A = [[i] for i in range(n)]
        B = [[i for i in range(n) if i != j] for j in range(n)]
        C = [[i for i in range(n)]]
        for a in A + B + C:
            if(self.func_calls+1 > self.T):
                break
            self.update(a,a,[i for i in range(n) if not i in a])

    def warm_up(self, o):
        n=self.game.n
        for s in range(2,n-1):
            pi = np.arange(n)
            np.random.shuffle(pi)
            for k in range(math.floor(n/s)):
                if(self.func_calls+1 > self.T):
                    break
                A = [pi[k*s+i] for i in range(s)]
                if o == "+":
                    self.update(A,A,[])
                else:
                    self.update([i for i in range(n) if not i in A], [], A)

            if n%s != 0:
                if(self.func_calls+1 > self.T):
                    break
                A = [pi[i] for i in range(n-(n%s),n)]
                l = s - (n%s)
                _B = np.arange(n)
                np.random.shuffle(_B)
                B = _B[:l]
                while set(B) == set(A):
                    _B = np.arange(n)
                    np.random.shuffle(_B)
                    B = _B[:l]
                if o == "+":
                    self.update([i for i in range(n) if i in A or i in B], A, [])
                else:
                    self.update([i for i in range(n) if not (i in A or i in B)], [], A)

    def p_size(self, s) -> int:
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

    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        self.phi_p = np.zeros((n,n))
        self.phi_m = np.zeros((n,n))
        self.c_p = np.zeros((n,n))
        self.c_m = np.zeros((n,n))
        if self.start_exact:
            self.exact_calculation()
        #self.warm_up("+")
        #self.warm_up("-")

        #assert self.func_calls == 2*n + 1 + 2*sum([math.ceil(n/s) for s in range(2,n-1)])
        while self.func_calls + 2 <= self.T:
            s = np.random.choice(np.arange(2,n-1), p=[self.p_size(s) for s in range(2,n-1)])
            A = np.arange(n)
            np.random.shuffle(A)
            A_t = list(A[:s])
            A_t1 = [i for i in range(n) if not i in A_t]
            self.update(A_t, A_t, [i for i in range(n) if not i in A_t])
            self.update(A_t1, A_t1, [i for i in range(n) if not i in A_t1])
            
        self.func_calls += 1
        self.save_steps(step_interval)


class StratSVARM(Algorithm):
    def __init__(self, start_exact=True, theoretical_distribution=True):
        self.start_exact = start_exact
        self.theoretical_distribution = theoretical_distribution

    def p_size(self, s) -> int:
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
        l = len(S)
        v = self.value(S)
        for player in S:
            self.phi_p[player, l-1] = (self.count_p[player, l-1] * self.phi_p[player, l-1] + v) / (self.count_p[player, l-1] + 1)
            self.count_p[player, l-1] += 1
        for player in notS:
            self.phi_m[player, l] = (self.count_m[player, l] * self.phi_m[player, l] + v) / (self.count_m[player, l] + 1)
            self.count_m[player, l] += 1
            
        self.phi = np.sum(self.phi_p-self.phi_m,axis=1)/self.game.n
        self.save_steps(self.step_interval)
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
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
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
            for player in range(n):
                not_player = np.concatenate((np.arange(player), np.arange(player+1, n)))
                self.update(not_player, np.array([player]))
                self.update(np.array([player]), not_player)
                
        while self.func_calls + 2 <= self.T:
            S, notS = self.sample()
            self.update(S, notS)
            self.update(notS, S)
            
        self.func_calls = self.T
        self.save_steps(step_interval)

class TruncStratSvarm(Algorithm):
    def __init__(self, trunc_l=0):
        self.trunc_l = trunc_l
    def update(self, A):
        l = len(A)-self.trunc_l
        n = self.game.n
        v = self.value(A)
        for i in A:
            self.phi_p[i, l-1] = (self.c_p[i, l-1] * self.phi_p[i, l-1] + v)/(self.c_p[i, l-1]+1)
            self.c_p[i, l-1] += 1

        for i in [i for i in range(n) if not i in A]:
            self.phi_m[i, l] = (self.c_m[i, l] * self.phi_m[i, l] + v)/(self.c_m[i, l]+1)
            self.c_m[i, l] += 1
        
        self.phi = 1/(n-self.trunc_l) * np.sum(self.phi_p-self.phi_m,axis=1)
        self.save_steps(self.step_interval)
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        self.phi_p = np.zeros((n,n-self.trunc_l))
        self.phi_m = np.zeros((n,n-self.trunc_l))
        self.c_p = np.zeros((n,n-self.trunc_l))
        self.c_m = np.zeros((n,n-self.trunc_l))
        self.phi_p[:, -1] = self.value(np.arange(n))
        self.c_p[:, -1] = 1

        while self.func_calls < self.T:
            s = np.random.choice(np.arange(self.trunc_l+1,n))
            A = np.arange(n)
            np.random.shuffle(A)
            A = A[:s]
            self.update(A)
            
        self.func_calls += 1
        self.save_steps(step_interval)
        
class BasicStratSVARM(Algorithm):
    def softmax(self, arr):
        exp = np.exp(arr)
        return exp/np.sum(exp)
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        self.phi_p = np.zeros((n,n))
        self.phi_m = np.zeros((n,n))
        self.count_p = np.zeros((n,n))
        self.count_m = np.zeros((n,n))
        topk = np.arange(k)
        v_all = self.value(np.arange(n))
        self.phi_p[:, -1] = v_all
        weights = np.ones(n-1)*k
        while self.func_calls + 1 < self.T:
            #probs = self.softmax(weights)
            #print(probs)
            strata = np.random.choice(np.arange(1,n))#, p = probs)
            S = np.arange(n)
            np.random.shuffle(S)
            S_not = S[strata:]
            S = S[:strata]
            
            value = self.value(S)
            self.phi_p[S, strata-1] = (self.phi_p[S, strata-1] * self.count_p[S, strata-1] + value) / (self.count_p[S, strata-1] + 1)
            self.count_p[S, strata-1] += 1
            
            self.phi_m[S_not, strata] = (self.phi_m[S_not, strata] * self.count_m[S_not, strata] + value) / (self.count_m[S_not, strata] + 1)
            self.count_m[S_not, strata] += 1
            
            self.phi = np.mean(self.phi_p-self.phi_m, axis=1)
            # topk_new = np.argpartition(self.phi, -k)[-k:]
            # # if(self.func_calls > 1):
            # #     weights[strata-1] += k - np.intersect1d(topk_new, topk).shape[0] - 1
            # topk = topk_new
            self.save_steps(step_interval)
            
        self.func_calls += 1
        self.save_steps(step_interval)