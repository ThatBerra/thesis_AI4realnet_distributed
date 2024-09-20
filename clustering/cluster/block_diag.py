import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.patches import Rectangle
import pandas as pd
import os
import grid2op

def dfs(adj_matrix, node, targets, variables, visited_r, visited_c, component, rc):
  #the argument 'rc' is checked because we need to act in different ways if we are considering a row or a column
  if rc == 'r':
    #keep track that the node is visited and append the variable it to the component we are updating
    visited_r[node] = True
    component.append(targets[node])

    #we iterate on the columns to check for possible neighbours
    for neighbour in range(adj_matrix.shape[1]):
      #if a non visited neighbour is found we call the recursive function on it
      if adj_matrix[node][neighbour] == True and not visited_c[neighbour]:
        dfs(adj_matrix, neighbour, targets, variables, visited_r, visited_c, component, rc='c')
  else:
    #same as before but keep in mind that we were passed a column
    visited_c[node] = True
    component.append(variables[node])

    for neighbour in range(adj_matrix.shape[0]):
      if adj_matrix[neighbour][node] == True and not visited_r[neighbour]:
        dfs(adj_matrix, neighbour, targets, variables, visited_r, visited_c, component, rc='r')
        

def find_connected_components(adj_matrix, targets, variables):
  n = adj_matrix.shape[0]
  m = adj_matrix.shape[1]
  components = []

  #define a visited flag for each target and for each variable
  visited_r = [False] * n
  visited_c = [False] * m

  for row in range(n):
    if not visited_r[row]:

      #init the component for the actual row as empty
      component = []

      #call the recursive function indicating that we are considering an element taken from rows (a target)
      dfs(adj_matrix, row, targets, variables, visited_r, visited_c, component, rc='r')
      components.append(component)

  return components


def block_diagonalization(matrix, targets, variables, thres):
    
    matrix_bin = matrix.copy()
    matrix_bin[matrix > thres] = 1
    matrix_bin[matrix <= thres] = 0
    
    components = find_connected_components(matrix_bin, targets, variables)
    
    rearranged_targets = []
    rearranged_variables = []
    for component in components:
        for n in component:
            if n in targets:
                rearranged_targets.append(n)
            else:
                rearranged_variables.append(n)
    df = pd.DataFrame(data=matrix_bin, index=targets, columns=variables)
                
    block_df = df.loc[rearranged_targets,rearranged_variables]
    
    return block_df, block_df.to_numpy(), components, rearranged_targets, rearranged_variables

def is_sparse(block):
    return np.sum(block) < block.size*0.5

def has_sparse_border(mat, row_offset, row_idx, col_offset, col_idx, border_type):
    if border_type == 'row':
        last_row = mat[row_idx-1,col_offset:col_idx]
        return np.sum(last_row) <= last_row.shape[0]*0.3
    elif border_type == 'col':
        last_column = mat[row_offset:row_idx,col_idx-1]
        return np.sum(last_column) <= last_column.shape[0]*0.3
    
def compute_metric(block):
    return np.sum(block) / block.size if block.size > 0 else 0

#TODO: Either clean or remove all of the following functions
def find_cliques(a):
    blocks_idx = []

    row_idx = 0
    col_idx = 0
    row_offset = 0
    col_offset = 0
    start_row_idx = -1
    start_col_idx = -1
    blocks_idx.append([row_offset, row_idx, col_offset, col_idx])

    j = 0
    while row_idx < a.shape[0]-1 and col_idx < a.shape[1]-1:
        
        # ---------------------------------------------------------
        # EXPAND BLOCK
        while row_idx > start_row_idx or col_idx > start_col_idx:

            start_row_idx = row_idx
            start_col_idx = col_idx

            # Move diagonal
            while row_idx < a.shape[0] and col_idx < a.shape[1]:
                if a[row_idx, col_idx] == 1:
                    row_idx += 1
                    col_idx += 1
                else:
                    break

            # Check down cell
            if row_idx < a.shape[0]:
                if a[row_idx,col_idx-1] == 1:
                    row_idx += 1
                    continue

            # Check right cell
            if col_idx < a.shape[1]:
                if a[row_idx-1,col_idx] == 1:
                    col_idx += 1
                
        block = a[row_offset:row_idx,col_offset:col_idx]

        # ---------------------------------------------------------------------    
        # REDUCE
        while is_sparse(block):  # if < 50% of elements are 1 reduce the block
            
            # Move left
            block = a[row_offset:row_idx,col_offset:col_idx-1]
            if block.size > 0 and not is_sparse(block):
                col_idx -= 1
                break

            # Move up
            block = a[row_offset:row_idx-1,col_offset:col_idx]
            if block.size > 0 and not is_sparse(block):
                row_idx -= 1
                break

            # Move diagonal
            row_idx -= 1
            col_idx -= 1
            block = a[row_offset:row_idx,col_offset:col_idx]

        if block.size == 0:  # restore to single 1 if reduction was not effective (row_idx == row_offset or col_idx == col_offset)
            if len(blocks_idx) > 0: # restore to single 1 if reduction was not effective (row_idx == row_offset or col_idx == col_offset)
                row_idx = blocks_idx[-1][1]
                col_idx = blocks_idx[-1][3]
            else:
                row_idx = 0
                col_idx = 0
            block = a[row_offset:row_idx,col_offset:col_idx]

        if np.sum(block) > 0:
            while has_sparse_border(a, row_offset, row_idx, col_offset, col_idx, border_type='row') or \
                    has_sparse_border(a, row_offset, row_idx, col_offset, col_idx, border_type='col'):
                
                if has_sparse_border(a, row_offset, row_idx, col_offset, col_idx, border_type='row'):
                    row_idx -= 1

                if has_sparse_border(a, row_offset, row_idx, col_offset, col_idx, border_type='col'):
                    col_idx -= 1

        # check if block has one element only
        block = a[row_offset:row_idx,col_offset:col_idx]

        while np.sum(block) <= 1:
            row_idx += 1
            col_idx += 1
            block = a[row_offset:row_idx,col_offset:col_idx]
            ones_idx = np.transpose(np.nonzero(block))
            if len(ones_idx) > 1:
                if row_idx > row_offset+3 and col_idx > col_offset+3:  
                    next_one_idx = ones_idx[1:][np.argmin(np.sum(ones_idx[1:], axis=1))]
                    row_idx = row_offset + next_one_idx[0] + 1
                    col_idx = col_offset + next_one_idx[1] + 1
                    row_offset = row_idx-1
                    col_offset = col_idx-1
                else:
                    next_one_idx = ones_idx[np.argmax(np.sum(ones_idx, axis=1))]  # priority to diagonal, then right, down
                    row_idx = row_offset + next_one_idx[0] + 1
                    col_idx = col_offset + next_one_idx[1] + 1

                    # Reshape previous block
                    if len(blocks_idx) > 0:
                        last_block = blocks_idx[-1].copy()
                        prev_row_offset = last_block[0]
                        prev_col_offset = last_block[2]

                        while has_sparse_border(a, prev_row_offset, row_idx, prev_col_offset, col_idx, border_type='row') or \
                                has_sparse_border(a, prev_row_offset, row_idx, prev_col_offset, col_idx, border_type='col'):
                            
                            if has_sparse_border(a, prev_row_offset, row_idx, prev_col_offset, col_idx, border_type='row'):
                                row_idx -= 1

                            if has_sparse_border(a, prev_row_offset, row_idx, prev_col_offset, col_idx, border_type='col'):
                                col_idx -= 1
                            if col_idx < 0:
                                break                

                        if row_idx == last_block[1] and col_idx == last_block[3]:  # if same block as before move to next one
                            row_idx = row_offset + next_one_idx[0] + 1
                            col_idx = col_offset + next_one_idx[1] + 1
                            row_offset = last_block[1]-1
                            col_offset = last_block[3]-1
                        else:
                            row_offset = last_block[0]
                            col_offset = last_block[2]
                            blocks_idx.pop()

                        # blocks_idx.pop()
            if row_idx == a.shape[0] or col_idx == a.shape[1]:
                break

        # -------------------------------------------------------------
        # EXPAND LAST BLOCK
            
        # Check presence of 1s in the last columns (backward)
        final_block = False
        if row_idx == a.shape[0]:
            final_block = True
            last_col_idx = a.shape[1]-1
            while last_col_idx > col_idx:
                if np.sum(a[row_offset:, last_col_idx]) > 0:
                    col_idx = last_col_idx+1
                    break
                last_col_idx -= 1

        # Check presence of 1s in the last rows (backward)
        if col_idx == a.shape[1]:
            final_block = True
            last_row_idx = a.shape[0]-1
            while last_row_idx >= row_idx:
                if np.sum(a[last_row_idx,col_offset:]) > 0:
                    row_idx = last_row_idx+1
                    break
                last_row_idx -= 1

        blocks_idx.append([row_offset, row_idx, col_offset, col_idx])

        block = a[row_offset:row_idx,col_offset:col_idx]
        for i in range(len(blocks_idx)):
            b = a[blocks_idx[i][0]:blocks_idx[i][1], blocks_idx[i][2]:blocks_idx[i][3]]
            if len(b) > 0:
                plt.figure()
                plt.title(f'Iter {j} | Block {i}: {blocks_idx[i]}')
                fm = sn.heatmap(data = b, annot=True, cbar=False)
                # plt.show(block=False)
                os.makedirs(os.path.join('fetch_data', 'iter'), exist_ok=True)
                plt.savefig(f'fetch_data/iter/{j}_b{i}.png', dpi=200)
                plt.close()

        # ---------------------------------------------------------------
        # set variables for next iter    
        
        # If reshaping of previous block produced no change, then force new block to start from next one
        # if expand_previous and blocks_idx[-1] == last_block:
        #     row_idx = next_one_idx[0]+1
        #     col_idx = next_one_idx[1]+1

        # Overlapping block (only last bottom right element)
        if not final_block:
            row_idx -= 1
            col_idx -= 1
            row_offset = row_idx
            col_offset = col_idx
            start_row_idx = row_idx-1
            start_col_idx = col_idx-1
            j+=1

    # ---------------------------------------------------------------
    # REFINE ---> maximize effectiveness by some metric

    # resolve overlapping borders (by construction all the borders are overlapping except when monoblock is found)

    for i in range(len(blocks_idx)-1):

        idx_1 = blocks_idx[i]
        idx_2 = blocks_idx[i+1]

        # check overlapping
        if idx_1[1]-1 == idx_2[0] and idx_1[3]-1 == idx_2[2]:  # row_idx[i]-1 == row_offset[i+1] and col_idx[i]-1 == col_offset[i+1]

            b1 = a[idx_1[0]:idx_1[1],idx_1[2]:idx_1[3]]
            b2 = a[idx_2[0]:idx_2[1],idx_2[2]:idx_2[3]]
    
            b1_diag = a[idx_1[0]:idx_1[1]-1,idx_1[2]:idx_1[3]-1]

            b1_left = a[idx_1[0]:idx_1[1],idx_1[2]:idx_1[3]-1]

            b1_up = a[idx_1[0]:idx_1[1]-1,idx_1[2]:idx_1[3]]

            b2_diag = a[idx_2[0]+1:idx_2[1],idx_2[2]+1:idx_2[3]]

            b2_right = a[idx_2[0]:idx_2[1],idx_2[2]+1:idx_2[3]]

            b2_down = a[idx_2[0]+1:idx_2[1],idx_2[2]:idx_2[3]]

            # all combo
            m_d1 = compute_metric(b1_diag) + compute_metric(b2)
            m_d2 = compute_metric(b1) + compute_metric(b2_diag)
            m_ld = compute_metric(b1_left) + compute_metric(b2_down)
            m_ur = compute_metric(b1_up) + compute_metric(b2_right)

            if m_d1 >= m_d2 and m_d1 >= m_ld and m_d1 >= m_ur:
                # reduce block 1 (decrease row, col idx)
                blocks_idx[i][1] -= 1
                blocks_idx[i][3] -= 1
            elif m_d2 >= m_d1 and m_d2 >= m_ld and m_d2 >= m_ur:
                # reduce block 2
                blocks_idx[i+1][0] += 1
                blocks_idx[i+1][2] += 1
            elif m_ld >= m_d1 and m_ld >= m_d2 and m_ld >= m_ur:
                blocks_idx[i][3] -= 1  # left
                blocks_idx[i+1][0] += 1  # down
            else:
                blocks_idx[i][1] -= 1  # up
                blocks_idx[i+1][2] += 1  # right

    # ----------------------------------------------------------------------
    # Minimum block size
                
    # first get rid of blocks of single 1 (evaluate borders)
    i = 0
    while i < len(blocks_idx):
        block_size = (blocks_idx[i][1]-blocks_idx[i][0])*(blocks_idx[i][3]-blocks_idx[i][2])
        if block_size == 1:
            previous_border_sum = 0.
            next_border_sum = 0.
            if i > 0:
                previous_border_sum = np.sum(a[blocks_idx[i][0], blocks_idx[i-1][2]:blocks_idx[i][3]]) + \
                      np.sum(a[blocks_idx[i-1][0]:blocks_idx[i][1],blocks_idx[i][2]])
            if i < len(blocks_idx)-1:
                next_border_sum = np.sum(a[blocks_idx[i][0], blocks_idx[i][2]:blocks_idx[i+1][3]]) + \
                        np.sum(a[blocks_idx[i][0]:blocks_idx[i+1][1],blocks_idx[i][2]])
            if previous_border_sum > next_border_sum:
                blocks_idx[i-1][1] = blocks_idx[i][1]
                blocks_idx[i-1][3] = blocks_idx[i][3]
            else:
                if i < len(blocks_idx)-1:
                    blocks_idx[i+1][0] = blocks_idx[i][0]
                    blocks_idx[i+1][2] = blocks_idx[i][2]
            blocks_idx.pop(i)
        else:
            i+=1

    # Then merge small blocks (evaluate metric)
    i = 0
    while i < len(blocks_idx):
        block_size = (blocks_idx[i][1]-blocks_idx[i][0])*(blocks_idx[i][3]-blocks_idx[i][2])
        if block_size <= 6:
            delta_score_previous = 0.
            delta_score_next = 0.
            # Merge to previous block
            if i > 0:
                previous_block = blocks_idx[i-1].copy()
                previous_block[1] = blocks_idx[i][1]
                previous_block[3] = blocks_idx[i][3]
                delta_score_previous = compute_metric(a[previous_block[0]:previous_block[1],previous_block[2]:previous_block[3]]) - \
                    compute_metric(a[blocks_idx[i-1][0]:blocks_idx[i-1][1],blocks_idx[i-1][2]:blocks_idx[i-1][3]])
            else:
                previous_block = [0, 0, 0, 0]
                previous_block[1] = blocks_idx[i][1]
                previous_block[3] = blocks_idx[i][3]
                delta_score_previous = compute_metric(a[previous_block[0]:previous_block[1],previous_block[2]:previous_block[3]]) - \
                    compute_metric(a[blocks_idx[i-1][0]:blocks_idx[i-1][1],blocks_idx[i-1][2]:blocks_idx[i-1][3]])        
            
            # Merge to next block
            if i < len(blocks_idx)-1:
                next_block = blocks_idx[i+1].copy()
                next_block[0] = blocks_idx[i][0]
                next_block[2] = blocks_idx[i][2]
                delta_score_next = compute_metric(a[next_block[0]:next_block[1], next_block[2]:next_block[3]]) - \
                    compute_metric(a[blocks_idx[i+1][0]:blocks_idx[i+1][1],blocks_idx[i+1][2]:blocks_idx[i+1][3]])
                    
            if delta_score_previous > delta_score_next:  # Take maximum score increment
                blocks_idx[i-1] = previous_block.copy()
            else:
                if i < len(blocks_idx) -1:
                    blocks_idx[i+1] = next_block.copy()
            blocks_idx.pop(i)
        else:
            i+=1

    return blocks_idx

def compute_total_score(blocks_idx, bm):
    total_score = 0
    for idx_list in blocks_idx:
      block = bm[idx_list[0]:idx_list[1],idx_list[2]:idx_list[3]]
      total_score += compute_metric(block)
    return total_score

def plot_results(bin, df, out_folder, quant):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sn.heatmap(data = bin, annot=True, cbar=False, ax=ax1)

    sn.heatmap(data = df, annot=True, cbar=False, ax=ax2)

    '''for i, idx_list in enumerate(blocks_idx):
        ax2.add_patch(Rectangle((idx_list[2], idx_list[0]), idx_list[3]-idx_list[2], idx_list[1]-idx_list[0],
                    edgecolor='red',
                    facecolor='none',
                    lw=2))
        print(f'Clique {i}: substations {list(df.columns)[idx_list[2]:idx_list[3]]}, lines {list(df.index)[idx_list[0]:idx_list[1]]}')'''


    #plt.suptitle(f'Quant = {quant}, score = {round(total_score,2)}', size=24)
    plt.suptitle(f'Quant = {quant}', size=24)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_folder, f'cliques_{quant}.png'), dpi=200)
    plt.close()

def diagonalize(a, path, quant_list, env_name='l2rpn_case14_sandbox'):
  os.makedirs(path, exist_ok=True)
  
  env = grid2op.make(env_name)
  idx = []
  n = env.observation_space.n_line
  m = env.observation_space.n_sub

  for sub in range(m):
      if env.observation_space.sub_info[sub] > 3:
          idx.append(sub)
  bin = np.zeros((n,m))

  targets = [f's{line}' for line in range(n)]
  variables = [f'sub{s}' for s in range(m)]

  for quant in quant_list:

    print()
    print(f'Threshold quantile: {quant}')

    for sub in idx:
        thresh = np.quantile(a[:,sub].flatten(), quant)
        bin[:,sub] = a[:,sub]>thresh

    bdf, bm, _, _, _ = block_diagonalization(bin, targets, variables, 0.75)
      
    #blocks_idx = find_cliques(bm)
    #total_score = compute_total_score(blocks_idx, bm)

    plot_results(bin, bdf, path, quant)

    #print(f'Score: {round(total_score,2)}')
    print()


def diagonalize_synthetic(a, path, thres_list):

  os.makedirs(path, exist_ok=True)
  
  n = a.shape[0]
  m = a.shape[1]

  targets = [f't{t}' for t in range(n)]
  variables = [f'v{v}' for v in range(m)]
  bin = np.zeros((n,m))

  for thres in thres_list:

    print()
    print(f'Threshold quantile: {thres}')

    for var in range(m):
        thresh = np.quantile(a[:,var].flatten(), thres)
        bin[:,var] = a[:,var]>thresh

    bdf, bm, _, _, _ = block_diagonalization(bin, targets, variables, thres)
    
    with open(os.path.join(path, f'bm_{thres}.npy'), 'wb') as f:
        np.save(f, bm)