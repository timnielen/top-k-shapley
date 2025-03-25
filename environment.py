from game import Game
from algorithms.algorithm import Algorithm, PAC_Algorithm
import numpy as np
import math
from tqdm import tqdm

class EvaluationEnvironment:
    def mse(self, phi, phi_estimated):
        '''
        returns the mean-squared error of the estimated shapley values

        Args:
            phi (float array of shape (n,)): the actual shapley values of the game
            phi_estimated (float array of shape (steps,n)): the estimated shapley values of the game for each step
        
        Returns:
            mse (array of shape (steps,)): the mean squared error at every step
        '''
        return np.mean((phi_estimated - phi)**2, axis=1)
    
    def epsilon_score(self, phi, border_player_value, top_k_estimated, rest_estimated):
        '''
        evaluates the epsilon required for the estimated solution to be epsilon approximate

        Args:
            phi (float array of shape (n,)): the actual shapley values of the game
            border_player_value (float): the kth largest shapley value
            top_k_estimated (int array of shape (steps, k)): the indices of the players identified as topk players at each step
            rest_estimated (int array of shape (steps, n-k)): the indices of the players identified as non-topk players at each step
        
        Returns:
            epsilon (array of shape (steps,)): the required epsilon>=0 at each step
        '''

        lower_epsilon = np.max(border_player_value - phi[top_k_estimated], axis=-1)
        upper_epsilon = np.max(phi[rest_estimated] - border_player_value, axis=-1)
        return np.max(np.concatenate((lower_epsilon[:, None], upper_epsilon[:, None]), axis=-1), axis=-1)
    
    def precision_score(self, relevant_players, candidates, top_k_estimated, k):
        '''
        evaluates the precision of the algorithm, i.e. whats the ratio of correctly identified topk players (ratio scale) 
        or did the algorithm find a correct solution (numeric scale)?

        Args:
            relevant_players (int array of shape (steps, h<k)): the players that are guaranteed to be part of the topk
            candidates (int array of shape (steps, l)): the players with a shapley value equal to the border, i.e. k-h of which can be chosen arbitrarily
            top_k_estimated (int array of shape (steps, k)): the indices of the players identified as topk players at each step
            k (int): the topk index
        
        Returns:
            ratio (float array of shape (steps,)): the ratio of correctly identified topk players at each step
            numeric (bool array of shape (steps,)): did the algorithm find a correct solution at each step
        '''
        num_correct = np.isin(top_k_estimated, relevant_players).sum(axis=1)
        num_correct += np.clip(np.isin(top_k_estimated, candidates).sum(axis=1), a_min = 0, a_max = k-relevant_players.shape[0])
        ratio = num_correct / k
        numeric = num_correct == k
        return ratio, numeric
    
    def average_measure(self, measure):
        '''averages a measure over a number of experiments
        
        Args:
            measure (float array of shape (num_experiments, steps)): the value of the measure in each experiment and at each step of the experiment
            
        Returns:
            average (float array of shape (steps,)): average of the measure at each experiment step
            se (float array of shape (steps,)): standard error of the average measure at each step obtained through the sample variance
        '''

        num_experiments, steps = measure.shape
        average = np.average(measure, axis=0)
        variance = np.sum((measure-average)**2, axis=0)/(num_experiments-1)
        se = np.sqrt(variance/num_experiments)
        return average, se


    def evaluate_fixed_k(self, game: Game, algorithm: Algorithm, k: int, budget: int, step_interval:int=100, num_experiments:int=100):
        '''
            evaluates an algorithm's performance on a cooperative game at different steps 
            and averages over multiple experiments to obtain the expected performance

            Returns:
                x (int array of shape (num_steps,)): the consumed budget at each step (for plotting purposes)
                result (dict): a dictionary containing the average and standard error of different measures ("ratio", "numeric", "epsilon", "mse") at each step 
        '''

        steps = budget // step_interval
        numeric_scale = np.zeros((num_experiments, steps))
        ratio_scale = np.zeros((num_experiments, steps))
        epsilons = np.zeros((num_experiments, steps))
        mse = np.zeros((num_experiments, steps))
        
        for experiment in range(num_experiments):
            game.initialize()
            algorithm.initialize(game, budget, step_interval)
            algorithm.get_top_k(k)
            assert len(algorithm.values) == steps, (len(algorithm.values), steps)
            
            phi = game.get_phi()
            phi_estimated = np.array(algorithm.values) 

            mse[experiment] = self.mse(phi, phi_estimated)
            sorted_estimated = np.argsort(-phi_estimated)
            top_k_estimated = sorted_estimated[:, :k]
            rest_estimated = sorted_estimated[:, k:]

            relevant_players, candidates, border = game.get_top_k(k) 
            epsilons[experiment] = self.epsilon_score(phi, border, top_k_estimated, rest_estimated)
            ratio_scale[experiment], numeric_scale[experiment] = self.precision_score(relevant_players, candidates, top_k_estimated, k)               

        result = {}
        result["ratio"] = self.average_measure(ratio_scale)
        result["numeric"] = self.average_measure(numeric_scale)
        result["mse"] = self.average_measure(mse)
        result["epsilon"] = self.average_measure(epsilons)
        x = (np.arange(steps)+1)*step_interval
        
        return x, result
    
    def evaluate_fixed_budget(self, game: Game, algorithm: Algorithm, K: np.ndarray, budget:int=1500, num_experiments:int=100):
        '''evaluates the algorithms performance using multiple values of K given a fixed budget

        Returns:
            K (array): returns the input K (for plotting purposes)
            result (dict): contains the average and standard error of each measure ("ratio", "numeric", "epsilon") 
        '''
        measures = ["ratio", "numeric", "epsilon"]
        result = {}
        for measure in measures:
            result[measure] = ([], [])
        with tqdm(K) as pbar:
            for k in pbar:
                _, sub_result = self.evaluate_fixed_k(game, algorithm, k, budget, step_interval=-1, num_experiments=num_experiments)
                for measure in measures:
                    avg, se = sub_result[measure]
                    result[measure][0] += [avg[-1]]
                    result[measure][1] += [se[-1]]
                pbar.set_postfix(k=k)
        for measure in measures:
            avg, se = result[measure]
            result[measure][0] = np.array(avg)
            result[measure][1] = np.array(se)
            assert np.array(avg).shape == (len(K),)
            assert np.array(se).shape == (len(K),)
        return K, result

    def evaluate_PAC(self, game: Game, algorithm: PAC_Algorithm, k: int, epsilon, num_experiments:int=100):
        '''evaluates the number of budget required for an algorithm to produce a probably approximatly correct solution averaged over multiple experiments
        
        Returns:
            average (float): average number of value function calls required
            se (float): standard error of the average obtained thrugh the sample variance
            accuracy (float): the actual ratio of approximately correct solutions
        '''
        func_calls = np.zeros(num_experiments, dtype=np.int32)
        num_correct = 0
        with tqdm(range(num_experiments)) as pbar:
            for round in pbar:
                n = game.n
                game.initialize(n)
                algorithm.initialize(game, budget=-1, step_interval=-1) #infinite budget
                algorithm.get_top_k(k)
                assert len(algorithm.values) == 1
                func_calls[round] = algorithm.func_calls
                
                phi = game.get_phi()
                sorted = np.argsort(-phi)
                top_k, rest = sorted[:k], sorted[k:]
                
                phi_estimated = algorithm.phi # same as algorithm.values[-1]
                sorted_estimated = np.argsort(-phi_estimated)
                top_k_estimated, rest_estimated = sorted_estimated[:k], sorted_estimated[k:]
                
                phi_k = phi[sorted][k-1]
                assert phi_k == np.min(phi[top_k])
                inclusion = np.min(phi[top_k_estimated]) >= phi_k - epsilon
                exclusion = np.max(phi[rest_estimated]) <= phi_k + epsilon
                num_correct += inclusion and exclusion
                
                pbar.set_postfix(topk_real=f"{np.sort(top_k)}", topk_approx=f"{np.sort(top_k_estimated)}", func_calls=f"{func_calls[round]}", accuracy=f"{num_correct/(round+1)}")
        
        avg, se = self.average_measure(func_calls)
        print(avg, se, num_correct/num_experiments)
        
        return avg, se, num_correct/num_experiments