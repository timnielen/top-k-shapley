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
        epsilons = np.zeros((rounds, steps))
        mse = np.zeros((rounds, steps))
        for i in range(rounds):
            game.initialize(n = self.n)
            algorithm.initialize(game, self.T)
            algorithm.get_top_k(k, step_interval)
            
            phi = np.array([game.get_phi(i) for i in range(self.n)])
            phi_estimated = np.array(algorithm.values)
            assert phi_estimated.shape == (steps, self.n), (phi_estimated.shape, (steps, self.n))
            mse[i] = np.sum((phi_estimated - phi)**2, axis=1)/self.n
            
            #epsilon score
            border_player_value = np.sort(phi)[-k]
            top_k_estimated = np.argsort(-phi_estimated)[:, :k]
            epsilons[i] = np.max(border_player_value - phi[top_k_estimated], axis=-1)
            assert np.all(epsilons[i] >= 0), (phi, border_player_value, top_k_estimated[-1])
            
            relevant_players, candidates, sum_topk = game.get_top_k(k) 
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
        variance_prec = np.sum((precisions-avg_prec)**2, axis=0)/(rounds-1)
        SE_prec = np.sqrt(variance_prec/rounds)
        
        avg_mse = np.average(mse, axis=0)
        variance_mse = np.sum((mse-avg_mse)**2, axis=0)/(rounds-1)
        SE_mse = np.sqrt(variance_mse/rounds)
        
        avg_percentage = np.average(percentage, axis=0)
        SE_percentage = np.sqrt(np.sum((percentage-avg_percentage)**2, axis=0)/(rounds-1))
        x = (np.arange(avg_prec.shape[0])+1)*step_interval
        
        avg_epsilon = np.average(epsilons, axis=0)
        variance_epsilon = np.sum((epsilons-avg_epsilon)**2, axis=0)/(rounds-1)
        SE_epsilon = np.sqrt(variance_epsilon/rounds)
        
        return x, avg_prec, SE_prec, avg_mse, SE_mse, avg_percentage, SE_percentage, avg_epsilon, SE_epsilon
    
    def evaluate_order(self, game: Game, algorithm: Algorithm, k: int, step_interval:int=100, rounds:int=100):
        steps = math.floor(self.T/step_interval)
        binary = np.zeros((rounds, steps))
        kendall = np.zeros((rounds, steps))
        spearman = np.zeros((rounds, steps))
        for i in range(rounds):
            game.initialize(n = self.n)
            algorithm.initialize(game, self.T)
            algorithm.get_top_k(k, step_interval)
            
            phi = np.array([game.get_phi(i) for i in range(self.n)])
            phi_estimated = np.array(algorithm.values)
            assert phi_estimated.shape == (steps, self.n), (phi_estimated.shape, (steps, self.n))
            
            correct_sorted = np.argsort(phi)
            estimated_sorted = np.argsort(phi_estimated)
            
            spearman[i] = 1 - 6*np.sum((correct_sorted-estimated_sorted)**2, axis=-1)/(self.n*(self.n**2 - 1))
            
            binary[i] = np.sum(correct_sorted == estimated_sorted, axis=-1)/self.n
            orders_real = np.tile(phi.reshape(-1, self.n, 1), (1,1,self.n)) > np.tile(phi, (1, self.n)).reshape(-1,self.n,self.n)
            orders_pred = np.tile(phi_estimated.reshape(-1, self.n, 1), (1,1,self.n)) > np.tile(phi_estimated, (1, self.n)).reshape(-1,self.n,self.n)
            num_discordant = np.sum(orders_real != orders_pred, axis=(-1,-2)) / 2
            kendall[i] = 1 - (4*num_discordant)/(self.n*(self.n-1))
                

        
        avg_binary = np.average(binary, axis=0)
        avg_kendall = np.average(kendall, axis=0)
        avg_spearman = np.average(spearman, axis=0)

        return avg_binary, avg_kendall, avg_spearman
    
    def get_numeric_scale(self, approximated: set, real: set):
        return approximated.issubset(real)
    
    def get_ratio_scale(self, approximated: set, real: set):
        return len(approximated.intersection(real))/len(approximated)
    
    def get_MSE(self, approximated, real):
        return (real-approximated)**2
    

class FixedBudgetEnvironment:
    def __init__(self, n: int, budget: int):
        self.budget = budget
        self.n = n
        
    def evaluate(self, game: Game, algorithm: Algorithm, K: np.ndarray, rounds:int=100):
        precisions = np.zeros((rounds, K.shape[0]))
        epsilons = np.zeros((rounds, K.shape[0]))
        for i in range(rounds):
            game.initialize(n = self.n)
            for index_k, k in enumerate(K):
                algorithm.initialize(game, self.budget)
                algorithm.get_top_k(k, step_interval=self.budget)
                
                phi = np.array([game.get_phi(i) for i in range(self.n)])
                phi_estimated = np.array(algorithm.values)
                top_k_estimated = np.argsort(-phi_estimated)[0, :k]
                
                relevant_players, candidates, sum_topk = game.get_top_k(k) 
                num_correct = np.isin(top_k_estimated, relevant_players).sum()
                num_correct += np.clip(np.isin(top_k_estimated, candidates).sum(), a_min = 0, a_max = k-relevant_players.shape[0])
                precisions[i, index_k] = num_correct/k
                
                
                #epsilon score
                border_player_value = np.sort(phi)[-k]
                epsilons[i, index_k] = np.max(border_player_value - phi[top_k_estimated], axis=-1)
                assert epsilons[i, index_k] >= 0, (phi, border_player_value, top_k_estimated[-1])
                
        avg_prec = np.average(precisions, axis=0)
        variance_prec = np.sum((precisions-avg_prec)**2, axis=0)/(rounds-1)
        SE_prec = np.sqrt(variance_prec/rounds)
        
        avg_epsilon = np.average(epsilons, axis=0)
        variance_epsilon = np.sum((epsilons-avg_epsilon)**2, axis=0)/(rounds-1)
        SE_epsilon = np.sqrt(variance_epsilon/rounds)
        
        return K, avg_prec, SE_prec, avg_epsilon, SE_epsilon