import pandas as pd
import numpy as np

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
    
    return block_df, block_df.to_numpy(), components

