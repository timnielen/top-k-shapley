# Antithetic Sampling for Top-k Shapley Identification

This repository contains the code for the paper **"Antithetic Sampling for Top-k Shapley Identification"**. The project explores algorithms for top-k feature importance in machine learning, using Shapley values and antithetic sampling techniques.

## Files

The main entry points for running the code are the following Python files:

- **`main.py`**: The main script to run the experiments given a maximum budget.
- **`pac.py`**: An additional script to run the experiments on PAC algorithms that stop once a PAC solution is found. 

## Setup

Before running the code, ensure you have the necessary Python dependencies. You can install them using `pip`

```bash
pip install -r requirements.txt
```

Ensure you have the required datasets available in the correct directories. You can modify the dataset paths directly in the files, or you can adjust them according to your specific needs.

## Configuration

### Parameters

At the beginning of both **`main.py`** and **`pac.py`**, you'll find a section where several important parameters are defined. You should modify these parameters based on your experiment setup.

## Running

Please note that running the scripts will take some additional time everytime a new game is evaluated for the first time. The reason is that the game values need to be reindexed and stored such that coalition values can be read efficiently. Local games take extra long as each subgame needs to be reindexed.