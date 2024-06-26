import numpy as np
import grid2op
import cmi_computation as cmi
np.random.seed(29)

if __name__=='__main__':

    path = 'case14_sub'
    env = grid2op.make('l2rpn_case14_sandbox')
    n = env.observation_space.n_line
    m = env.observation_space.n_sub
    connections = env.action_space.sub_info
    
    for sub in range(m):
        if connections[sub] > 3:
            history = np.load(f'{path}{sub}_hist.npz')['data']
            np.random.shuffle(history[:,:20])

            mi_matrix, eta = cmi.compute_mi_matrix_parallel(n, m, i, history)

            with open(f'{path}{sub}_shuffledMI.npy', 'wb') as f:
                np.save(f, mi_matrix)

            with open(f'{path}{sub}_shuffledMI_time', 'w')  as f:
                f.write(str(eta))

