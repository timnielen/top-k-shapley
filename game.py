import numpy as np
import random
import pandas as pd
import math
import os

class Game:
    def initialize(self, n: int):
        self.n = n
    def get_phi(self, i: int) -> float:
        pass
    def value(self, S: list) -> float:
        pass
    def get_top_k(self, k: int):
        assert k > 0
        phi = np.array([self.get_phi(i) for i in range(self.n)])
        sorted = np.argsort(-phi)
        border = phi[sorted[k-1]]
        relevant_players = sorted[:k-1][phi[sorted[:k-1]] > border]
        candidates = sorted[k-1:][phi[sorted[k-1:]] == border]
        sum_topk = np.sum(phi[sorted[:k]])
        return relevant_players, candidates, sum_topk


class ShoeGame(Game):
    def get_phi(self, i: int) -> float:
        return 1/2
    
    def value(self, S: list) -> float:
        left = np.array([s for s in S if s%2==0])
        right = np.array([s for s in S if s%2==1])
        return min(len(left), len(right))

class SymmetricVotingGame(Game):
    def get_phi(self, i: int) -> float:
        return 1/self.n
    
    def value(self, S: list) -> float:
        if(len(S) > (self.n/2)):
            return 1
        return 0
    
class NonSymmetricVotingGame(Game):
    def __init__(self, w: list = [45,41,27,26,26,25,21,17,17,14,13,13,12,12,12,11,10,10,10,10,9,9,9,9,8,8,7,7,7,7,6,6,6,6,5,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3]):
        self.w = w
        phi = np.zeros(len(w))
        phi[:10] = [0.08831, 0.07973, 0.05096, 0.04898, 0.04898, 0.04700, 0.03917, 0.03147, 0.03147, 0.02577]
        phi[10:12], phi[12:15], phi[15], phi[16:20] = 0.02388, 0.02200, 0.02013, 0.01827
        phi[20:24], phi[24:26], phi[26:30], phi[30:34] =  0.01641, 0.01456, 0.01272, 0.01088
        phi[34], phi[35:44], phi[44:51] =  0.009053, 0.007230, 0.005412
        print(phi)
        self.phi = phi

    def get_phi(self, i: int) -> float:
        assert self.n == len(self.w)
        return self.phi[i]
    
    def value(self, S: list) -> float:
        assert self.n == len(self.w)
        if(sum([self.w[i] for i in S]) > (sum(self.w)/2)):
            return 1
        return 0

class AirportGame(Game):
    def __init__(self, c: list = [1 for i in range(8)] \
                 + [2 for i in range(12)] \
                 + [3 for i in range(6)] \
                 + [4 for i in range(14)] \
                 + [5 for i in range(8)] \
                 + [6 for i in range(9)] \
                 + [7 for i in range(13)] \
                 + [8 for i in range(10)] \
                 + [9 for i in range(10)] \
                 + [10 for i in range(10)], \
                  s: list = [0.01 for i in range(8)] \
                 + [0.020869565 for i in range(12)] \
                 + [0.033369565 for i in range(6)] \
                 + [0.046883079 for i in range(14)] \
                 + [0.063549745 for i in range(8)] \
                 + [0.082780515 for i in range(9)] \
                 + [0.106036329 for i in range(13)] \
                 + [0.139369662 for i in range(10)] \
                 + [0.189369662 for i in range(10)] \
                 + [0.289369662 for i in range(10)]):
        self.c = c
        self.s = s
        self.name = ""

    def get_phi(self, i: int) -> float:
        assert self.n == len(self.c)
        return self.s[i]
    
    def value(self, S: list) -> float:
        assert self.n == len(self.c)
        if(len(S) == 0):
            return 0
        return max([self.c[i] for i in S])

class UnanimityGame(Game):
    def initialize(self, n: int):
        self.n = n
        R = np.arange(n)
        np.random.shuffle(R)
        self.R = R[:random.randint(1, n+1)]

    def get_phi(self, i: int) -> float:
        if(i in self.R):
            return 1/len(self.R)
        return 0
    
    def value(self, S: list) -> float:
        if(set(self.R).issubset(set(S))):
            return 1
        return 0
    
class SumUnanimityGames(Game):
    def __init__(self, min_games=5, max_games=50):
        self.min_games = min_games
        self.max_games = max_games
        self.name = ""

    def initialize(self, n: int):
        num_games = random.randint(self.min_games, self.max_games)
        self.R = [UnanimityGame() for _ in range(num_games)]
        self.calls = 0

        self.n = n
        for game in self.R:
            game.initialize(n)
        self.c = [random.random()/len(game.R) for game in self.R]

         
    def get_phi(self, i: int) -> float:
        return sum([self.c[index] * game.get_phi(i) for index, game in enumerate(self.R)])
    
    def value(self, S: list) -> float:
        self.calls += 1
        return sum([self.c[index] * game.value(S) for index, game in enumerate(self.R)])

def binom(n, k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))


def index_to_coalition(index):
    view = np.array([index]).view(np.uint8)
    return np.where(np.unpackbits(view, bitorder='little'))[0]

def coalition_to_index(coalition):
    if(coalition.shape[0] == 0):
        return 0
    return np.sum(1 << coalition)


class GlobalFeatureImportance(Game):
    def __init__(self, filepath, num_players, use_cached=True):
        self.n = num_players
        self.name = filepath.split('/')[-1]
        values_path = f"{filepath.split('.')[0]}_values.npy"
        shapley_values_path = f"{filepath.split('.')[0]}_shapley_values_phi.npy"
        if use_cached:
            try:
                self.values = np.load(values_path)
            except:
                print(f"could not find cached values. manual reindexing...")
                self.df = pd.read_csv(filepath)
                self.values = self.reindex()
                np.save(values_path, self.values)
            
            try:
                self.phi = np.load(shapley_values_path)
            except:
                print(f"could not find cached shapley values. manual calculation...")
                self.phi = self.exact_calculation()
                np.save(shapley_values_path, self.phi)
                
        else: 
            self.values = self.reindex()
            np.save(values_path, self.values)
            self.phi = self.exact_calculation()
            np.save(shapley_values_path, self.phi)
        assert num_players == np.log2(self.values.shape[0])
        print(self.values)
        print(self.phi, np.sum(self.phi))
        
    def value(self, S):
        return self.values[coalition_to_index(np.array(S))]
    def reindex(self):
        values = np.zeros((2**self.n))
        for index, row in self.df.iterrows():
            if index == 0:
                continue
            coalition = np.array(row["coalition"].split('|'), dtype=int)
            coalition_index = coalition_to_index(coalition)
            values[coalition_index] = row["value"]
            if index%1000 == 0:
                print(index)
        return values
    def exact_calculation(self):
        weights = np.zeros(self.n)
        for length in range(self.n):
            weights[length] = 1/(self.n*binom(self.n-1, length))
        
        phi = np.zeros(self.n)
        for index in range(2**self.n):
            coalition = index_to_coalition(index)
            length = coalition.shape[0]
            for player in range(self.n):
                if player in coalition:
                    phi[player] += weights[length-1] * self.values[index]
                else:
                    phi[player] -= weights[length] * self.values[index]
        return phi
    def get_phi(self, i: int) -> float:
        return self.phi[i]
    
class LocalFeatureImportance(Game):
    def initialize(self, n):
        games = os.listdir(self.directory)
        games = [filename for filename in games if filename.split(".")[-1] == "csv"]
        game = np.random.choice(games)
        # game = "688.csv"
        # print(game)
        filepath = f"{self.directory}/{game}"
        self.n = n
        values_path = f"{filepath.split('.')[0]}_values.npy"
        shapley_values_path = f"{filepath.split('.')[0]}_shapley_values_phi.npy"
        if self.use_cached:
            try:
                self.values = np.load(values_path)
            except:
                print(f"could not find cached values. manual reindexing...")
                self.df = pd.read_csv(filepath)
                self.values = self.reindex()
                np.save(values_path, self.values)
            
            try:
                self.phi = np.load(shapley_values_path)
            except:
                print(f"could not find cached shapley values. manual calculation...")
                self.phi = self.exact_calculation()
                np.save(shapley_values_path, self.phi)
                
        else: 
            self.values = self.reindex()
            np.save(values_path, self.values)
            self.phi = self.exact_calculation()
            np.save(shapley_values_path, self.phi)
        
        assert n == np.log2(self.values.shape[0])
        # print(self.values)
        # print(self.phi, np.sum(self.phi))
    def __init__(self, directory, num_players, use_cached=True):
        self.directory = directory
        self.n = num_players
        self.use_cached = use_cached
        self.name = directory.split('/')[-1]
        
    def reindex(self):
        values = np.zeros((2**self.n))
        num_sets_per_length = np.array([binom(self.n, l) for l in range(self.n+1)])
        min_index_per_length = [np.sum(num_sets_per_length[:l]) for l in range(self.n+1)]
        min_index_per_length
        for i in range(2**self.n):
            coalition = index_to_coalition(i)
            name = f"s{''.join(coalition.astype('str'))}"
            rows = self.df[self.df["set"] == name]
            if rows.shape[0] > 1:
                rows = rows[rows.index >= min_index_per_length[coalition.shape[0]]]
                rows = rows[rows.index < min_index_per_length[coalition.shape[0]+1]]
            val = rows["value"]
            values[i] = val.to_numpy()[0]
            if i%1000 == 0:
                print(i)
        return values
    
    
    def value(self, S):
        return self.values[coalition_to_index(np.array(S))]
    def exact_calculation(self):
        weights = np.zeros(self.n)
        for length in range(self.n):
            weights[length] = 1/(self.n*binom(self.n-1, length))
        
        phi = np.zeros(self.n)
        for index in range(2**self.n):
            coalition = index_to_coalition(index)
            length = coalition.shape[0]
            for player in range(self.n):
                if player in coalition:
                    phi[player] += weights[length-1] * self.values[index]
                else:
                    phi[player] -= weights[length] * self.values[index]
        return phi
    def get_phi(self, i: int) -> float:
        return self.phi[i]

class UnsupervisedFeatureImportance(GlobalFeatureImportance):
    def reindex(self):
        values = np.zeros((2**self.n))
        num_sets_per_length = np.array([binom(self.n, l) for l in range(self.n+1)])
        min_index_per_length = [np.sum(num_sets_per_length[:l]) for l in range(self.n+1)]
        min_index_per_length
        for i in range(2**self.n):
            coalition = index_to_coalition(i)
            name = f"s{''.join(coalition.astype('str'))}"
            rows = self.df[self.df["set"] == name]
            if rows.shape[0] > 1:
                rows = rows[rows.index >= min_index_per_length[coalition.shape[0]]]
                rows = rows[rows.index < min_index_per_length[coalition.shape[0]+1]]
            val = rows["value"]
            values[i] = val.to_numpy()[0]
            if i%1000 == 0:
                print(i)
        return values
    def exact_calculation(self):
        weights = np.zeros(self.n)
        for length in range(self.n):
            weights[length] = 1/(self.n*binom(self.n-1, length))
        
        phi = np.zeros(self.n)
        for index in range(2**self.n):
            coalition = index_to_coalition(index)
            length = coalition.shape[0]
            for player in range(self.n):
                if player in coalition:
                    phi[player] += weights[length-1] * self.values[index]
                else:
                    phi[player] -= weights[length] * self.values[index]
        return phi