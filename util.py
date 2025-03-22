import math
import numpy as np

def binom(n, k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))


def index_to_coalition(index):
    '''
    converts an index to a coalition using its binary representation
    E.g.: 
        binary(0) = 0000 -> coalition []
        binary(1) = 0001 -> coalition [0]
        binary(2) = 0010 -> coalition [1]
        binary(5) = 0101 -> coalition [0,2]
    '''
    view = np.array([index]).view(np.uint8)
    return np.where(np.unpackbits(view, bitorder='little'))[0]

def coalition_to_index(coalition):
    '''
    converts a coalition to an index using its binary representation
    E.g.: 
        coalition [] -> 0000 = binary(0) -> 0
        coalition [0] -> 0001 = binary(1) -> 1
        coalition [1] -> 0010 = binary(2) -> 2
        coalition [0,2] -> 0101 = binary(5) -> 5
    '''
    if(coalition.shape[0] == 0):
        return 0
    return np.sum(1 << coalition)

def clear_ith_bit(number, i):
    # Use bitwise AND with a mask where the ith bit is 0, and all others are 1
    return number & ~(1 << i)

def set_ith_bit(number, i):
    # Use bitwise OR with a mask where only the ith bit is set to 1
    return number | (1 << i)

def calc_variance(n, values, phi):
    num_games, num_coalitions = values.shape
    weights = np.zeros(n+1)
    for length in range(n+1):
        weights[length] = 1/((n+1)*binom(n, length))
    
    # calc E[X^2]
    variance = np.zeros((num_games, n))
    for index in range(2**n):
        coalition = index_to_coalition(index)
        length = coalition.shape[0]
        weight = weights[length]
        player_coalitions = np.array([[set_ith_bit(index, player), clear_ith_bit(index, player)] for player in range(n)]).transpose()
        marginals = values[:, player_coalitions[0]] - values[:, player_coalitions[1]]
        variance += weight * (marginals**2)
    
    # calc E[X^2] - E[X]^2
    variance -= phi**2
    return variance

def calc_covariance(n, values, phi):
    num_games, num_coalitions = values.shape
    weights = np.zeros(n+1)
    for length in range(n+1):
        weights[length] = 1/((n+1)*binom(n, length))
        
    covariances = np.zeros((num_games, n,n))
    ## calc E[XY]
    for index in range(2**n):
        coalition = index_to_coalition(index)
        length = coalition.shape[0]
        weight = weights[length]
        player_coalitions = np.array([[set_ith_bit(index, player), clear_ith_bit(index, player)] for player in range(n)]).transpose()
        marginals = values[:, player_coalitions[0]] - values[:, player_coalitions[1]]
        covariances += weight * (marginals[:, :, np.newaxis] * marginals[:, np.newaxis, :])
    
    ## calc E[X]E[Y]
    covariances -= phi[:, :, np.newaxis] * phi[:, np.newaxis, :]
        
    return covariances