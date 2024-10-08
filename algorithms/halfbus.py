from .algorithm import Algorithm
import numpy as np
import math

class HalfBUS(Algorithm):
    def __init__(self, focus=1, trunc_l=0, trunc_v=0): #focus=w
        self.focus = focus
        self.trunc_l = trunc_l
        self.trunc_v = trunc_v
    def value(self, S: list):
        if len(S) < self.trunc_l:
            return self.trunc_v
        assert self.func_calls < self.T
        self.func_calls += 1
        v = self.game.value(S)
        return v
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        t = np.zeros(n)
        v_n = self.value(np.arange(n))
        def sample():
            length = np.random.choice(np.arange(n+1))
            S = np.arange(n)
            np.random.shuffle(S)
            return S[:length]
        while self.func_calls+2 <= self.T:
            pi = np.argsort(-self.phi)
            border = (self.phi[pi[k-1]] + self.phi[pi[k]])/2
            delta = np.abs(self.phi - border) * t
            P = np.argsort(delta)[:self.focus]
            S = sample()
            l = len(S)
            if l == 0:
                v = 0
            elif l == n:
                v = v_n
            else:
                v = self.value(S)
            for p in P:
                if self.func_calls == self.T:
                    return
                if p in S:
                    marginal = v - self.value(S[S != p])
                else:
                    marginal = (self.value(np.concatenate((S, [p]))) - v)
                self.phi[p] = (t[p]*self.phi[p]+marginal)/(t[p]+1)
                t[p] += 1
                self.save_steps(step_interval)
        if self.func_calls < self.T:
            self.func_calls += 1
        self.save_steps(step_interval)
        

def binom(n, k):
    # try:
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    # except:
    #     print(n, k)
    #     return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

class SIRHalfBUS(Algorithm):
    def sample(self, dont=np.array([])):
        length = np.random.choice(np.arange(self.game.n+1))
        S = np.setdiff1d(np.arange(self.game.n), dont)
        np.random.shuffle(S)
        return S[:length]
    def get_value(self, S):
        l = len(S)
        if l == 0:
            v = 0
        elif l == self.game.n:
            v = self.v_n
        else:
            v = self.value(S)
        return v
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        t = np.zeros(n)
        self.v_n = self.value(np.arange(n))
        pre_samples = 100
        total_weights = 0
        radius = 2
        num_query_players = 2*radius
        for l in range(n+1):
            for w in range(max(0, l-n+num_query_players), min(num_query_players, l)+1):
                total_weights += (w+1) * binom(num_query_players, w) * binom(n-num_query_players, l-w)
                # except:
                #     print(l,w)
                #     total_weights += (w+1) * binom(num_query_players, w) * binom(n-num_query_players, l-w)
        coalition_weights = np.array([1/((n+1)*binom(n, l)) for l in range(n+1)])
        while self.func_calls+1 < self.T:
            p_k = np.argsort(-self.phi)[k-radius:k+radius]
            lengths = np.random.randint(n+1, size=(pre_samples))
            samples = np.mgrid[0:pre_samples, 0:n][1]
            for i in range(pre_samples):
                np.random.shuffle(samples[i])
                samples[i, lengths[i]:] = -1
            #print(samples)
            pdf = (np.isin(samples, p_k).sum(axis=1)+1) / total_weights
            weights = pdf #/ coalition_weights[lengths]
            selected_coalition_index = np.random.choice(np.arange(pre_samples), p=weights/np.sum(weights))
            length = lengths[selected_coalition_index]
            S = samples[selected_coalition_index, :length]
            coalition_weight = pdf[selected_coalition_index]
            
            v = self.get_value(S)
            for p in range(n):
                if self.func_calls == self.T:
                    self.save_steps(step_interval)
                    return
                if p in S:
                    marginal = v - self.get_value(S[S != p])
                    # marginal *= coalition_weights_in[len(S)] / (currWeight/total_weights)
                else:
                    marginal = self.get_value(np.concatenate((S, [p]))) - v
                    # marginal *= coalition_weights_out[len(S)] / (currWeight/total_weights)
                # marginal *=  coalition_weights[length] / coalition_weight
                marginal /=  coalition_weight
                self.phi[p] = (t[p]*self.phi[p]+marginal)/(t[p]+1)
                t[p] += 1
                self.save_steps(step_interval)
        if self.func_calls < self.T:
            self.func_calls += 1
        self.save_steps(step_interval)
        
    
            
class StratHalfBUS(Algorithm):
    def __init__(self, focus=1, start_exact=False, trunc_l=0):
        self.focus = focus
        self.start_exact = start_exact
        self.trunc_l = trunc_l
    def update(self, S, f=["+", "-"], P=[]): 
        if self.func_calls+1 > self.T:
            return
        n = self.game.n
        l = len(S)
        if(l == 0):
            v=0
        if(l == n):
            v = self.v_n
        else:
            v = self.value(S)
        
        plus = np.zeros(n)
        minus = np.zeros(n)
        for i in P:
            if i in S:
                plus[i] = 1
            else:
                minus[i] = 1
        plus = plus == 1
        minus = minus == 1
        
        if("+" in f):
            self.phi_p[plus, l-1] = (self.phi_p[plus, l-1] * self.c_p[plus, l-1] + v) / (self.c_p[plus, l-1] + 1)
            self.c_p[plus, l-1] += 1
        if("-" in f and l != n):
            self.phi_m[minus, l] = (self.phi_m[minus, l] * self.c_m[minus, l] + v) / (self.c_m[minus, l] + 1)
            self.c_m[minus, l] += 1
        
        self.phi = np.sum(self.phi_p/(np.sum(self.c_p>0, axis=1)+1) - self.phi_m/(np.sum(self.c_m>0, axis=1)+1),axis=1)
        self.save_steps(self.step_interval)

    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        self.phi_p = np.zeros((n,n-self.trunc_l))
        self.phi_m = np.zeros((n,n-self.trunc_l))
        self.c_p = np.zeros((n,n-self.trunc_l))
        self.c_m = np.zeros((n,n-self.trunc_l))
        self.v_n = self.value(np.arange(n))
        def sample():
            if(self.start_exact):
                length = np.random.choice(np.arange(2,n-1))
            else:
                length = np.random.choice(np.arange(self.trunc_l+1,n+1))
            #S = np.concatenate((np.arange(i), np.arange(i+1, n)))
            S = np.arange(n)
            np.random.shuffle(S)
            return S[:length]
        
        A = [[i] for i in range(n)] #l=1
        B = [[i for i in range(n) if i != j] for j in range(n)] #l=n-1
        C = [i for i in range(n)] #l=n
        if(self.start_exact):
            for a in A + [C]:
                if(self.func_calls+1 > self.T):
                    break
                P = np.arange(n)
                self.update(a, f=["+"], P=P)
            for i in range(n):
                self.update([], f=["-"], P=[i])
            for index, a in enumerate(B):
                assert not index in a
                self.update(a, f=["-"], P=[index])
            t = 2 * np.ones(n)
        else:
            t = np.zeros(n)
            
        while self.func_calls+self.focus+1 <= self.T:
            pi = np.argsort(-self.phi)
            border = (self.phi[pi[k-1]] + self.phi[pi[k]])/2
            delta = np.abs(self.phi - border) * t
            P = np.argsort(delta)[:self.focus]
            for p in P:
                for i in [j for j in range(n) if not j in P]:
                    assert delta[p] <= delta[i], (delta, p, i)
            t[P] += 1
            S = sample()
            l = len(S)
            if l == 0:
                h = 0
            elif l == n:
                h = self.v_n
            else:
                h = self.value(S)
            for p in P:
                if p in S:
                    plus = h
                    minus = self.value(S[S != p])
                    l = len(S) - 1 - self.trunc_l
                else:
                    plus = self.value(np.concatenate((S, [p])))
                    minus = h
                    l = len(S)-self.trunc_l
                self.phi_p[p, l] = (self.phi_p[p, l] * self.c_p[p, l] + plus) / (self.c_p[p, l] + 1)
                self.c_p[p, l] += 1
                self.phi_m[p, l] = (self.phi_m[p, l] * self.c_m[p, l] + minus) / (self.c_m[p, l] + 1)
                self.c_m[p, l] += 1
                self.phi = 1/n * np.sum(self.phi_p - self.phi_m,axis=1)
                self.save_steps(self.step_interval)
                
        while self.func_calls < self.T:
            self.func_calls += 1
            self.save_steps(step_interval)

class HalfBUSAVG(Algorithm):
    def __init__(self, focus=1, start_exact=False, trunc_l=0, trunc_v=0):
        self.focus = focus
        self.start_exact = start_exact
        self.trunc_l = trunc_l
        self.trunc_v = trunc_v
    def value(self, S: list):
        if len(S) < self.trunc_l:
            return self.trunc_v
        assert self.func_calls < self.T
        self.func_calls += 1
        v = self.game.value(S)
        return v
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        t = np.zeros(n)
        v_n = self.value(np.array([i for i in range(n)]))
        def sample():
            length = np.random.choice(np.arange(n+1))
            S = np.arange(n)
            np.random.shuffle(S)
            return S[:length]
        while self.func_calls+2 <= self.T:
            pi = np.argsort(-self.phi)
            border = (self.phi[pi[k-1]] + self.phi[pi[k]])/2
            delta = np.abs(self.phi - border) * t
            P = np.argsort(delta)[:self.focus]
            S = sample()
            l = len(S)
            if l == 0:
                v = 0
            elif l == n:
                v = v_n
            else:
                v = self.value(S)
            avg_in = 0
            avg_out = 0
            for p in P:
                if self.func_calls == self.T:
                    return
                if p in S:
                    marginal = v - self.value(S[S != p])
                    avg_in += marginal
                else:
                    marginal = (self.value(np.concatenate((S, [p]))) - v)
                    avg_out += marginal
                self.phi[p] = (t[p]*self.phi[p]+marginal)/(t[p]+1)
                t[p] += 1
                self.save_steps(step_interval)
            l_in = len([p for p in P if p in S])
            if l_in == 0:
                l_in = 1
            l_out = len([p for p in P if not p in S])
            if l_out == 0:
                l_out = 1
            avg_in /= l_in
            avg_out /= l_out
            for p in [p for p in range(n) if not p in P]:
                if p in S:
                    self.phi[p] = (t[p]*self.phi[p]+avg_in)/(t[p]+1)
                else:
                    self.phi[p] = (t[p]*self.phi[p]+avg_out)/(t[p]+1)
                t[p] += 1
        if self.func_calls < self.T:
            self.func_calls += 1
        self.save_steps(step_interval)
        