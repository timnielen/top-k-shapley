import math
import numpy as np

def binom(n, k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))


def index_to_coalition(index):
    view = np.array([index]).view(np.uint8)
    return np.where(np.unpackbits(view, bitorder='little'))[0]

def coalition_to_index(coalition):
    if(coalition.shape[0] == 0):
        return 0
    return np.sum(1 << coalition)

def clear_ith_bit(number, i):
    # Use bitwise AND with a mask where the ith bit is 0, and all others are 1
    return number & ~(1 << i)

def set_ith_bit(number, i):
    # Use bitwise OR with a mask where only the ith bit is set to 1
    return number | (1 << i)