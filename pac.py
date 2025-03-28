from algorithms.dict import algorithms
from algorithms.algorithm import PAC_Algorithm
from environment import EvaluationEnvironment
import numpy as np
from game import GlobalFeatureImportance, LocalFeatureImportance, UnsupervisedFeatureImportance
from plot import plot
import os
from tqdm import tqdm
import pandas as pd

#################################################
#          CHANGE PARAMETERS AS NEEDED          #
#################################################

GAME_KIND = "global" # "local", "unsupervised"
GAME_PATH_GLOBAL = "datasets/global/Bank marketing classification random forest.csv" # path to csv file
GAME_PATH_UNSUPERVISED = "datasets/unsupervised/vf_BigFive.csv" # path to csv file
GAME_PATH_LOCAL = "datasets/local/image_classification" # path to directory containing csv files
USE_CACHED=True # use cached (shapley) values of the games for faster evaluation if possible

TEST_ALGORITHMS = ["SamplingSHAP@K", "CMCS", "CMCS@K", "Greedy CMCS"]
K=3
NUM_EXPERIMENTS=2
T_MIN = 30
DELTA = 0.01
EPSILON = 0.0005

DATA_DIRECTORY="results/pac"

#################################################
#                                               #
#################################################

if __name__ == "__main__":
    env = EvaluationEnvironment()
    
    if GAME_KIND == "global":
        game = GlobalFeatureImportance(GAME_PATH_GLOBAL, use_cached=USE_CACHED)
    elif GAME_KIND == "local":
        game = LocalFeatureImportance(GAME_PATH_LOCAL, use_cached=USE_CACHED)
    elif GAME_KIND == "unsupervised":
        game = UnsupervisedFeatureImportance(GAME_PATH_UNSUPERVISED, use_cached=USE_CACHED)
    else: 
        raise ValueError(f'Unknown game kind: "{GAME_KIND}"!')
    
    results = []
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    
    test_algorithms = TEST_ALGORITHMS
    if len(test_algorithms) == 0:
        test_algorithms = algorithms.keys()
        
    with tqdm(test_algorithms) as pbar:
        for name in pbar:
            pbar.set_postfix_str(name)
            if name not in algorithms.keys():
                raise ValueError(f'Unknown algorithm: "{name}"!')
            algorithm = algorithms[name]

            assert K > 0 and K < game.n
            assert issubclass(algorithm, PAC_Algorithm)
            results.append((name, *env.evaluate_PAC(game=game, algorithm=algorithm(t_min=T_MIN, delta=DELTA, epsilon=EPSILON), k=K, epsilon=EPSILON, num_experiments=NUM_EXPERIMENTS)))
            pbar.set_postfix(accuracy=results[-1][-1])
        
        df = pd.DataFrame(results, columns=["algorithm", "num_calls_avg", "num_calls_se", "accuracy"])
        # for i, result in enumerate(results):
        #     df.iloc[i] = result
        dir = os.path.join(DATA_DIRECTORY, f"k={K}_rounds={NUM_EXPERIMENTS}", GAME_KIND)
            
        if not os.path.isdir(dir):
            os.makedirs(dir)
            
        df.to_csv(os.path.join(dir, f"{game.name}.csv"), index=False)
            
        