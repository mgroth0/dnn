import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib
matplotlib.rcParams["backend"] = 'Agg'
if __name__ == '__main__':
    with open('problem_array.pkl', 'rb') as f:
        problem_array = pickle.load(f)
    print(f'len:{len(problem_array)}')
    print(f'min:{min(problem_array)}')
    print(f'max:{max(problem_array)}')
    print(f'num NaNs:{len([n for n in problem_array if np.isnan(n)])}')
    print(f'num floats:{len([n for n in problem_array if isinstance(n,float)])}')
    print(f'backend:{matplotlib.rcParams["backend"]}')
    plt.violinplot([problem_array])
    plt.show()
