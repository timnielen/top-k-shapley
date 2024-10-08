from .algorithm import Algorithm
import numpy as np
import math

def C(n, k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

class ISSV(Algorithm):
    def get_top_k(self, k: int, step_interval: int = 100):
        n = self.game.n
        self.phi = np.zeros(n)
        t = np.zeros(n)
        complex_weights = [[] for i in range(n)]
        simple_weights = [[] for i in range(n)]
        badAvg = np.zeros(n)
        badt = np.zeros(n)

        while self.func_calls + 2 <= self.T:
            for i in range(n):
                length = np.random.choice(np.arange(n))
                S = np.concatenate((np.arange(i), np.arange(i+1, n)))
                np.random.shuffle(S)
                S = S[:length]
                if self.func_calls + 2 > self.T:
                    break
                marginal = self.value(np.concatenate((S, [i]))) - self.value(S)
                simple_weights[i].append(1/(n*C(n-1,length)))
                complex_weights[i].append(marginal)
                self.phi[i] = 0
                t[i] = 0
                for _ in range(8):
                    complexPDF = np.array(complex_weights[i])
                    scomplex = np.sum(complexPDF)
                    if scomplex == 0: 
                        complexPDF = np.ones_like(complexPDF)
                    complexPDF /= np.sum(complexPDF)
                    simplePDF = np.array(simple_weights[i])
                    simplePDF /= np.sum(simplePDF)
                    ##print(complexPDF)
                    y = np.random.choice(np.arange(len(complexPDF)), p=np.array(complexPDF))
                    e = complex_weights[i][y]
                    w_e = np.sum(complexPDF/simplePDF)/(len(complexPDF)*complexPDF[y])
                    self.phi[i] = (t[i]*self.phi[i]+w_e*e)/(t[i]+1)
                    t[i] += 1
                self.save_steps(step_interval)
        if self.func_calls < self.T:
            self.func_calls += 1
        self.save_steps(step_interval)