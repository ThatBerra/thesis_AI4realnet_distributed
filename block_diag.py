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
      if adj_matrix[node][neighbour] == 1 and not visited_c[neighbour]:
        dfs(adj_matrix, neighbour, targets, variables, visited_r, visited_c, component, rc='c')
  else:
    #same as before but keep in mind that we were passed a column
    visited_c[node] = True
    component.append(variables[node])

    for neighbour in range(adj_matrix.shape[0]):
      if adj_matrix[neighbour][node] == 1 and not visited_r[neighbour]:
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


if __name__=='__main__':
    
    with open('cmi_large.npy', 'rb') as f:
        cmi_matrix = np.load(f)
        
    targets = ['s1_','s2_','s3_','s4_','s5_']
    variables = ['s1','s2','s3','s4','s5','a1','a2','a3']
    df = pd.DataFrame(data=cmi_matrix, index=targets, columns=variables)
    
    thresh = 0.01
    cmi_matrix_bin = cmi_matrix
    cmi_matrix_bin[cmi_matrix > thresh] = 1
    cmi_matrix_bin[cmi_matrix <= thresh] = 0
    print(cmi_matrix_bin)
    
    components = find_connected_components(cmi_matrix_bin, targets, variables)
    print(components)

