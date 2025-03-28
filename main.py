from algorithms.dict import algorithms
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

TEST_ALGORITHMS = ["CMCS", "compSHAP"]
K=3 # if K=-1 eval all k with fixed budget
BUDGET=1500
NUM_EXPERIMENTS=500
STEP_INTERVAL=50 # budget interval between datapoints 

PLOT=True
PLOT_SAVE=False
PLOT_DIRECTORY="results/plots" # where the plots should be saved
MEASURES = ["ratio", "epsilon", "mse"] # the measures to be plotted

DATA=True # save the resulting data as csv
DATA_DIRECTORY="results/data"

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
            
            if K>0 and K<game.n:
                results.append((name, *env.evaluate_fixed_k(game=game, algorithm=algorithm(), k=K, budget=BUDGET, step_interval=STEP_INTERVAL, num_experiments=NUM_EXPERIMENTS)))
            elif K==-1:
                results.append((name, *env.evaluate_fixed_budget(game=game, algorithm=algorithm(), K=np.arange(1,game.n), budget=BUDGET, num_experiments=NUM_EXPERIMENTS)))
            else:
                raise ValueError("Invalid K!")
            
    if PLOT:
        if K>0 and K<=game.n:
            path = os.path.join(PLOT_DIRECTORY, "fixed_k", f"{GAME_KIND}({game.name})_k={K}_budget={BUDGET}_rounds={NUM_EXPERIMENTS}.pdf")
            plot(results, "budget", MEASURES, PLOT_SAVE, path)
        elif K==-1:
            path = os.path.join(PLOT_DIRECTORY, "fixed_budget", f"{GAME_KIND}({game.name})_budget={BUDGET}_rounds={NUM_EXPERIMENTS}.pdf")
            plot(results, "k", MEASURES, PLOT_SAVE, path)
        else:
            raise ValueError("Invalid K!")
        
    if DATA:
        for name, x, result in results:
            df = pd.DataFrame()
            xlabel = "budget" if K!=-1 else "K"
            klabel = K if K!=-1 else f"1-{game.n-1}"
            df[xlabel] = x
            dir = os.path.join(DATA_DIRECTORY, GAME_KIND, game.name, f"budget={BUDGET}_rounds={NUM_EXPERIMENTS}", f"k={klabel}")
            
            for measure, (avg, se) in result.items():
                df[measure] = avg
                df[f"{measure}_SE_plus"] = avg + se
                df[f"{measure}_SE_minus"] = avg - se
                
            if not os.path.isdir(dir):
                os.makedirs(dir)
                
            df.to_csv(os.path.join(dir, f"{name}.csv"), index=False)
            
        