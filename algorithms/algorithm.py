import numpy as np

class Algorithm:
    def initialize(self, game, budget: int, save_intermediate = False):
        self.values = []
        self.game = game
        self.T = budget
        self.func_calls = 0
        self.phi = np.zeros(game.n)
        self.save_intermediate = save_intermediate
        if self.save_intermediate:
            np.save(f"tmp/values_{self.__class__.__name__}_{self.func_calls}.npy", self.values)

    def get_top_k(self, k: int, step_interval: int = 100):
        pass

    def save_steps(self, step_interval: int):
        if(self.func_calls/step_interval >= len(self.values) + 1):
            self.values += [np.array(self.phi)]
            if self.save_intermediate:
                np.save(f"tmp/values_{self.__class__.__name__}_{step_interval}_{self.func_calls}.npy", self.values)

    def value(self, S: list):
        assert self.func_calls < self.T
        self.func_calls += 1
        return self.game.value(S)
    
class QuickSelectShapley(Algorithm):
    def compare(self, p1: int, p2: int, budget: int):
        n = self.game.n
        assert p1 != p2
        fak = 1
        if(p1 > p2):
            p1, p2 = p2, p1
            fak = -1
        def sample(p1: int, p2: int) -> list:
            assert p1 < p2
            length = np.random.choice(np.arange(n-1))
            S = np.concatenate((np.arange(p1), np.arange(p1+1, p2), np.arange(p2+1,n)))
            np.random.shuffle(S)
            return list(S[:length])
        
        
        prev_calls = self.func_calls
        
        diff=0
        t=0
        while(self.func_calls-prev_calls+2 <= budget and self.func_calls+2 <= self.T):
            S = sample(p1, p2)
            t += 1
            assert not p1 in S and not p2 in S
            diff_new = (self.value(S + [p1]) - self.value(S + [p2]))
            diff = ((t-1)*diff+diff_new)/t

        return fak * diff
    
    def get_top_k(self, k: int, step_interval: int = 100):
        max = self.T
        for i in range(math.floor(max/step_interval)):
            self.func_calls = 0
            self.T = step_interval * (i+1)
            self.values += [self._get_top_k(k)]
            
            
    def _get_top_k(self, k: int):
        n = self.game.n
        def partition(l, left, right, pivotIndex):
            pivot = l[pivotIndex]
            l[pivotIndex], l[right] = l[right], l[pivotIndex]
            save_index = left
            for i in range(left, right):
                if (self.compare(l[i], pivot, self.T/(4*n)) < 0):
                    l[save_index], l[i] = l[i], l[save_index]
                    save_index += 1
            l[right], l[save_index] = l[save_index], l[right]    
            return save_index
        
        def choose(l, left, right, k):
            if(left == right):
                return l[k:]
            pivotIndex = np.random.choice(np.arange(left, right+1))
            pivotIndex = partition(l, left, right, pivotIndex)
            if(k == pivotIndex):
                return l[k:]
            elif(k<pivotIndex):
                return choose(l, left, pivotIndex - 1, k)
            return choose(l, pivotIndex + 1, right, k)
        
        topk = choose(np.arange(n), 0, n-1, n-k)
        assert len(topk) == k
        return np.array([1 if i in topk else 0 for i in range(n) ])
    
    
class FakeSvarm(Algorithm):
    def update(self, S, p=None): 
        n = self.game.n
        v = self.value(S)
        l = len(S)
        bias = 1
        for i in S:
            if p != None and p!=i:
                if p in S:
                    bias = (n-1)/(l-1)
                else:
                    bias = (n-1)/(n-l)
            self.phi_p[i, l-1] = (self.c_p[i, l-1] * self.phi_p[i, l-1] + v/bias)/(self.c_p[i, l-1]+1)
            self.c_p[i, l-1] += 1
        for i in [i for i in range(n) if not i in S]:
            if p != None and p!=i:
                if p in S:
                    bias = (n-1)/l
                else:
                    bias = (n-1)/(n-l-1)
            self.phi_m[i, l] = (self.c_m[i, l] * self.phi_m[i, l] + v/bias)/(self.c_m[i, l]+1)
            self.c_m[i, l] += 1
        
        self.phi = 1/self.game.n * np.sum(self.phi_p-self.phi_m,axis=1)
        self.save_steps(self.step_interval)

    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        self.phi_p = np.zeros((n,n))
        self.phi_m = np.zeros((n,n))
        self.c_p = np.zeros((n,n))
        self.c_m = np.zeros((n,n))
        t = np.zeros(n)
        def sample(i):
            length = np.random.choice(np.arange(1,n))
            S = np.concatenate((np.arange(i), np.arange(i+1,n)))
            np.random.shuffle(S)
            return list(S[:length])
        
        A = [[i] for i in range(n)] #l=1
        B = [[i for i in range(n) if i != j] for j in range(n)] #l=n-1
        C = [[i for i in range(n)]] #l=n
        for a in C:# + B + A:
            self.update(a)
        
        while self.func_calls+2 <= self.T:
            #pi = np.argsort(-self.phi)
            #border = (self.phi[pi[k-1]] + self.phi[pi[k]])/2
            #delta = [abs(self.phi[i] - border) for i in range(n)]
            #i = np.argmin([delta[j] * t[j] for j in range(n)])
            #t[i] += 1
            for i in range(n):
                if self.func_calls+2 > self.T:
                    break
                S = sample(i)
                self.update(S + [i])
                self.update(S)
            
        while self.func_calls < self.T:
            self.func_calls += 1
            self.save_steps(step_interval)




class FakeSvarm2(Algorithm):
    def update(self, S): 
        n = self.game.n
        v = self.value(S)
        l = len(S)
        plus = np.array([1 if i in S else 0 for i in range(n)]) == 1
        minus = plus == 0
        self.phi_p[plus, l-1] = (self.phi_p[plus, l-1] * self.c_p[plus, l-1] + v)
        self.c_p[plus, l-1] += 1
        self.phi_p[plus, l-1] /= self.c_p[plus, l-1]

        if(l!=n):
            self.phi_m[minus, l] = (self.phi_m[minus, l] * self.c_m[minus, l] + v)
            self.c_m[minus, l] += 1
            self.phi_m[minus, l] /= self.c_m[minus, l]
        
        self.phi = np.sum(self.phi_p/np.sum(self.c_p>0, axis=1) - self.phi_m/np.sum(self.c_m>0, axis=1),axis=1)
        self.save_steps(self.step_interval)

    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        self.phi_p = np.zeros((n,n))
        self.phi_m = np.zeros((n,n))
        self.c_p = np.zeros((n,n))
        self.c_m = np.zeros((n,n))
        def sample():
            length = np.random.choice(np.arange(2,n-1))
            S = np.arange(n)
            np.random.shuffle(S)
            return list(S[:length])
        
        A = [[i] for i in range(n)] #l=1
        B = [[i for i in range(n) if i != j] for j in range(n)] #l=n-1
        C = [[i for i in range(n)]] #l=n
        for a in A + B + C:
            if(self.func_calls+1 > self.T):
                break
            self.update(a)

        # for s in range(2,n-1):
        #     pi = np.arange(n)
        #     np.random.shuffle(pi)
        #     if not self.func_calls+1 <= self.T:
        #         return
        #     self.update(pi[:s])
        
        while self.func_calls+1 <= self.T:
            S = sample()
            self.update(S)