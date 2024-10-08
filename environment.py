from game import Game
from algorithms.algorithm import Algorithm
import numpy as np
import math

class Environment:
    def __init__(self, n: int, budget: int, metric="ratio"):
        self.T = budget
        self.n = n
        self.metric = metric

    def evaluate(self, game: Game, algorithm: Algorithm, k: int, step_interval:int=100, rounds:int=100):
        steps = math.floor(self.T/step_interval)
        precisions = np.zeros((rounds, steps))
        mse = np.zeros((rounds, steps))
        for i in range(rounds):
            game.initialize(n = self.n)
            algorithm.initialize(game, self.T)
            algorithm.get_top_k(k, step_interval)
            real = np.array([game.get_phi(i) for i in range(self.n)])
            approximated = np.array(algorithm.values)
            top_k_real = set(game.get_top_k(k))
            top_k_approximated = np.argpartition(algorithm.values, -k)[:, -k:]
            assert top_k_approximated.shape == (steps, k), (top_k_approximated.shape, (steps, k))
            assert approximated.shape == (steps, self.n), (approximated.shape, (steps, self.n))
            mse[i] = np.array([sum(self.get_MSE(step, real))/self.n for step in approximated])
            if self.metric == "ratio":
                precisions[i] = [self.get_ratio_scale(set(step), top_k_real) for step in top_k_approximated]
            else:
                precisions[i] = [self.get_numeric_scale(set(step), top_k_real) for step in top_k_approximated]

        
        avg_prec = np.average(precisions, axis=0)
        avg_mse = np.average(mse, axis=0)
        variance_prec = np.sum((precisions-avg_prec)**2, axis=0)/(rounds-1)
        variance_mse = np.sum((mse-avg_mse)**2, axis=0)/(rounds-1)
        SE_prec = np.sqrt(variance_prec/rounds)
        SE_mse = np.sqrt(variance_mse/rounds)

        return avg_prec, SE_prec, avg_mse, SE_mse

    def get_numeric_scale(self, approximated: set, real: set):
        return approximated.issubset(real)
    
    def get_ratio_scale(self, approximated: set, real: set):
        return len(approximated.intersection(real))/len(approximated)
    
    def get_MSE(self, approximated, real):
        return (real-approximated)**2
