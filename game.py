import numpy as np
import random
import pandas as pd
import math
import os
from util import *

class Game:
    def initialize(self):
        '''should be run infront of each experiment in case the selected game contains randomness 
        (such as sampling a random local game in each experiment)'''
        pass
    def get_phi(self) -> np.ndarray:
        return self.phi
    def value(self, S) -> float:
        pass
    def get_top_k(self, k: int):
        '''returns two partitions of players:
            1. relevant_players: set of h<k players whose shapley values are larger than the border, i.e. must be part of the topk
            2. candidates: set of all players whose shapley values are equal to the border, any subset of k-h of these may be part of the topk
        and returns the border value, i.e. the value of the candidates
        '''
        assert k > 0
        phi = self.get_phi()
        sorted = np.argsort(-phi)
        border = phi[sorted[k-1]]
        relevant_players = sorted[:k-1][phi[sorted[:k-1]] > border]
        candidates = sorted[k-1:][phi[sorted[k-1:]] == border]
        return relevant_players, candidates, border
    
    def reindex(self):
        '''given the dataframe obtained from the game's csv file compute an array containing all coalition values
        where the binary representation of the index is indicative of the coalition.
        E.g.: 
            binary(0) = 0000 -> coalition []
            binary(1) = 0001 -> coalition [0]
            binary(2) = 0010 -> coalition [1]
            binary(5) = 0101 -> coalition [0,2]
        '''
        pass
    
    def exact_calculation(self):
        '''
        - calculate exact shapley values for each player using all values.
        - assumes values to be reindexed
        '''
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

class GlobalFeatureImportance(Game):
    '''uses precomputed coalition values stored in a csv document'''
    def __init__(self, filepath, use_cached=True):
        self.name = filepath.split('/')[-1].split('.')[0]
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
            self.df = pd.read_csv(filepath)
            self.values = self.reindex()
            np.save(values_path, self.values)
            self.phi = self.exact_calculation()
            np.save(shapley_values_path, self.phi)

        self.n = np.log2(self.values.shape[0])
        print(self.values, self.n)
        print(self.phi, np.sum(self.phi))
        
    def value(self, S):
        return self.values[coalition_to_index(np.array(S))]
    def reindex(self):
        '''given the dataframe obtained from the game's csv file compute an array containing all coalition values
        where the binary representation of the index is indicative of the coalition.
        E.g.: 
            binary(0) = 0000 -> coalition []
            binary(1) = 0001 -> coalition [0]
            binary(2) = 0010 -> coalition [1]
            binary(5) = 0101 -> coalition [0,2]
        '''
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
        
    
class LocalFeatureImportance(Game):
    '''uses precomputed coalition values stored in csv documents'''
    def __init__(self, directory, use_cached=True):
        self.directory = directory
        self.use_cached = use_cached
        self.name = directory.split('/')[-1]

    def initialize(self):
        '''initializes the game's values by sampling a random local game and reading the corresponding values from disk.
        Then, reindex the values for quick access
        '''
        games = os.listdir(self.directory)
        games = [filename for filename in games if filename.split(".")[-1] == "csv"]
        game = np.random.choice(games)
        filepath = f"{self.directory}/{game}"
        values_path = f"{filepath.split('.')[0]}_values.npy"
        shapley_values_path = f"{filepath.split('.')[0]}_shapley_values_phi.npy"
        if self.use_cached:
            try:
                self.values = np.load(values_path)
            except:
                print(f"could not find cached values. manual reindexing...")
                self.df = pd.read_csv(filepath)
                self.values = self.reindex(self.df)
                np.save(values_path, self.values)
            
            try:
                self.phi = np.load(shapley_values_path)
            except:
                print(f"could not find cached shapley values. manual calculation...")
                self.phi = self.exact_calculation(self.values)
                np.save(shapley_values_path, self.phi)
                
        else: 
            self.values = self.reindex(self.df)
            np.save(values_path, self.values)
            self.phi = self.exact_calculation(self.values)
            np.save(shapley_values_path, self.phi)
        
        self.n = np.log2(self.values.shape[0])
        
    def reindex(self, df):
        '''given the dataframe obtained from the game's csv file compute an array containing all coalition values
        where the binary representation of the index is indicative of the coalition.
        E.g.: 
            binary(0) = 0000 -> coalition []
            binary(1) = 0001 -> coalition [0]
            binary(2) = 0010 -> coalition [1]
            binary(5) = 0101 -> coalition [0,2]
        '''
        values = np.zeros((2**self.n))
        num_sets_per_length = np.array([binom(self.n, l) for l in range(self.n+1)])
        min_index_per_length = [np.sum(num_sets_per_length[:l]) for l in range(self.n+1)]
        min_index_per_length
        for i in range(2**self.n):
            coalition = index_to_coalition(i)
            name = f"s{''.join(coalition.astype('str'))}"
            rows = df[df["set"] == name]
            if rows.shape[0] > 1:
                rows = rows[rows.index >= min_index_per_length[coalition.shape[0]]]
                rows = rows[rows.index < min_index_per_length[coalition.shape[0]+1]]
            val = rows["value"]
            values[i] = val.to_numpy()[0]
            if i%1000 == 0:
                print(i)
        return values
    
    def reindex_all(self):
        '''reindexes all games' values contained in the game directory (see reindex for details)'''
        games = [filename for filename in os.listdir(self.directory) if filename.split(".")[-1] == "csv"]
        game_paths = [f"{self.directory}/{filename}" for filename in games]
        value_paths = [f"{filepath.split('.')[0]}_values.npy" for filepath in game_paths]
        values = np.zeros((len(game_paths), 2**self.n))
        for index, path in enumerate(value_paths):
            if self.use_cached:
                try:
                    values[index] = np.load(path)
                except:
                    print(f"could not find cached values. manual reindexing...")
                    df = pd.read_csv(game_paths[index])
                    values[index] = self.reindex(df)
                    np.save(path, values[index])
            else:
                df = pd.read_csv(game_paths[index])
                values[index] = self.reindex(df)
                np.save(path, values[index])
        return values, games
    
    def get_all_phi(self, all_values):
        '''calculate the shapley values of all players for each local subgame's csv file  '''
        games = [filename for filename in os.listdir(self.directory) if filename.split(".")[-1] == "csv"]
        game_paths = [f"{self.directory}/{filename}" for filename in games]
        shapley_values_paths = [f"{filepath.split('.')[0]}_shapley_values_phi.npy" for filepath in game_paths]
        shapley_values = np.zeros((len(game_paths), self.n))
        for index, path in enumerate(shapley_values_paths):
            if self.use_cached:
                try:
                    shapley_values[index] = np.load(path)
                except:
                    print(f"could not find cached shapley values. manual reindexing...")
                    df = pd.read_csv(game_paths[index])
                    shapley_values[index] = self.exact_calculation(df, all_values[index])
                    np.save(path, shapley_values[index])
            else:
                df = pd.read_csv(game_paths[index])
                shapley_values[index] = self.exact_calculation(df, all_values[index])
                np.save(path, shapley_values[index])
        return shapley_values, games
    
    def value(self, S):
        return self.values[coalition_to_index(np.array(S))]

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