import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
np.seterr(all='raise')
matplotlib.rcParams["backend"] = 'Agg'
if __name__ == '__main__':
    with open('problem_array_4.pkl', 'rb') as f:
        problem_array = pickle.load(f)

    # z = np.zeros((len(problem_array),))
    # for i, p in enum(problem_array):
    #     # assert isinstance(p, np.float64)
    #     z[i] = float(p)
    # problem_array = z
    # problem_array = np.array(problem_array)

    print(f'type:{type(problem_array)}')
    print(f'shape:{(problem_array)}')
    print(f'min:{min(problem_array)}')
    print(f'max:{max(problem_array)}')
    print(f'num NaNs:{len([n for n in problem_array if np.isnan(n)])}')
    print(f'num floats:{len([n for n in problem_array if isinstance(n, float)])}')
    print(f'backend:{matplotlib.rcParams["backend"]}')
    plt.violinplot([problem_array])
    plt.show()


    # File('problem_array_3.pkl').save(problem_array[0])
    # from lib.rsa_figs import RSAViolinBuilder
    # data: RSAViolinBuilder = File(
    #     '/Users/matt/Desktop/registered/todo/science/dnn/freecomp/data/result/rsa/rsa/33/res/violin-L2-Norm-width-RN18-200.pkl').load()
    # problem_array = list(data.sas.data.values())
    # File('problem_array_2.pkl').save(problem_array)

    # breakpoint()
