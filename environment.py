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
        percentage = np.zeros((rounds, steps))
        mse = np.zeros((rounds, steps))
        for i in range(rounds):
            game.initialize(n = self.n)
            algorithm.initialize(game, self.T)
            algorithm.get_top_k(k, step_interval)
            
            phi = np.array([game.get_phi(i) for i in range(self.n)])
            phi_estimated = np.array(algorithm.values)
            assert phi_estimated.shape == (steps, self.n), (phi_estimated.shape, (steps, self.n))
            mse[i] = np.sum((phi_estimated - phi)**2, axis=1)/self.n
            
            relevant_players, candidates, sum_topk = game.get_top_k(k) 
            top_k_estimated = np.argpartition(algorithm.values, -k)[:, -k:]
            assert top_k_estimated.shape == (steps, k), (top_k_estimated.shape, (steps, k))
            num_correct = np.isin(top_k_estimated, relevant_players).sum(axis=1)
            num_correct += np.clip(np.isin(top_k_estimated, candidates).sum(axis=1), a_min = 0, a_max = k-relevant_players.shape[0])
            
            # print(relevant_players, candidates, top_k_estimated[-1], num_correct[-1])
            
            if self.metric == "ratio":
                precisions[i] = num_correct/k
            else:
                precisions[i] = num_correct == k
            
            percentage[i] = np.sum(phi[top_k_estimated], axis=1) / sum_topk
                

        
        avg_prec = np.average(precisions, axis=0)
        avg_mse = np.average(mse, axis=0)
        variance_prec = np.sum((precisions-avg_prec)**2, axis=0)/(rounds-1)
        variance_mse = np.sum((mse-avg_mse)**2, axis=0)/(rounds-1)
        SE_prec = np.sqrt(variance_prec/rounds)
        SE_mse = np.sqrt(variance_mse/rounds)
        avg_percentage = np.average(percentage, axis=0)
        SE_percentage = np.sqrt(np.sum((percentage-avg_percentage)**2, axis=0)/(rounds-1))

        return avg_prec, SE_prec, avg_mse, SE_mse, avg_percentage, SE_percentage

    def get_numeric_scale(self, approximated: set, real: set):
        return approximated.issubset(real)
    
    def get_ratio_scale(self, approximated: set, real: set):
        return len(approximated.intersection(real))/len(approximated)
    
    def get_MSE(self, approximated, real):
        return (real-approximated)**2
