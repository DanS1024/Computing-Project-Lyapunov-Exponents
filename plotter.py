import matplotlib.pyplot as plt
import numpy as np

def plot(name, res, var, save=False, folder=''):
    with open('LCEs.txt', 'a') as f:
        print(f'\nLyapunov exponents for the {name}:')
        f.write(f'\nLyapunov exponents for the {name}:\n')
        plt.figure()
        for i in range(len(var)):
            print(f'{res[2][i]} +- {res[3][i]}')
            f.write(f'{res[2][i]} +- {res[3][i]}\n')

            plt.plot(res[0][i][:], label=f'{res[0][i][-1]:.6f}')
        
        print(f'KY dimension: {res[4]}')
        f.write(f'KY dimension: {res[4]}\n')


        plt.title(f'Lyapunov exponents for the {name}')
        plt.xlabel('Iteration')
        plt.ylabel('Lyapunov exponents')
        plt.legend()
        if save: plt.savefig(f'{folder}{name} LCEs.png')

    if len(var) == 2: # 2D phase space
        plt.figure()
        plt.scatter(res[1][0], res[1][1], c=np.arange(len(res[1][0])), s=0.5)
        plt.xlabel(str(var[0]))
        plt.ylabel(str(var[1]))
        plt.title(f'Phase space trajectory of the {name}')
        plt.colorbar(label='Iteration')
        if save: plt.savefig(f'{folder}{name}.png')
    elif len(var) >= 3: # projected 3D phase space
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(res[1][0], res[1][1], res[1][2], c=np.arange(len(res[1][0])), s=0.5)
        ax.set_xlabel(str(var[0]))
        ax.set_ylabel(str(var[1]))
        ax.set_zlabel(str(var[2]))
        ax.set_title(f'Phase space trajectory of the {name}')
        if save: plt.savefig(f'{folder}{name}.png')