from math import ceil
import numpy as np
import sys
from itertools import chain
import time
from pysat.solvers import Glucose3
def seconds_separator(seconds_passed):
  duration = seconds_passed
  hours = int(duration//3600)
  duration = duration%3600
  minutes = int(duration//60)
  duration = float(round(duration%60, 4))
  return str(hours)+":"+"0"*(minutes<10)+str(minutes)+":"+"0"*(duration<10)+str(duration)+"0"*(6-len(str(duration))+(duration>=10))
np.set_printoptions(threshold=sys.maxsize)

def create_array(values:int=1, n_variables:int=1) -> np.array:
  return np.array([0]*(abs(values)-1)+[(values>0)*2-1]+[0]*(n_variables-abs(values)), dtype=np.int8).reshape(1,-1)

def delete_zeros(matrix:np.array, n_variables:int) -> np.array:
  return matrix[np.where((matrix == 0).sum(axis=1) != n_variables)[0]]

def simplify_cnf_formula(matrix:np.array, n_variables:int) -> np.array:
  simplified = matrix.copy()
  for x in range(n_variables-1, 0, -1):
    with_x_zeros = np.where((simplified == 0).sum(axis=1) == x)[0]
    if (with_x_zeros.size != 0) and ((simplified == 0).sum(axis=1).min() < x):
      comparables = simplified[with_x_zeros]
      erasables = simplified[np.where((simplified == 0).sum(axis=1) < x)[0]]
      with_more_zeros = np.where((simplified == 0).sum(axis=1) > x)[0]
      if (with_more_zeros.size != 0):
        saveables = simplified[with_more_zeros]
        for clause in comparables: erasables = erasables[np.where((erasables*(clause!=0) != clause).sum(axis=1) > 0)[0]]
        simplified = np.vstack([saveables, comparables, erasables])
      else:
        for clause in comparables: erasables = erasables[np.where((erasables*(clause!=0) != clause).sum(axis=1) > 0)[0]]
        simplified = np.vstack([comparables, erasables])
  return simplified

def conjunction_cnfs(matrix1:np.array, matrix2:np.array, n_variables:int) -> np.array:
  return simplify_cnf_formula(np.unique(np.vstack([matrix1, matrix2]), axis=0), n_variables)

def disjunction_cnfs(matrix1:np.array, matrix2:np.array, n_variables:int) -> np.array:
  new_ones = []
  for clause in matrix1:
    new_ones += [delete_zeros((matrix2+clause-clause*(matrix2==clause))*((matrix2*clause < 0).sum(axis=1) == 0).reshape(-1,1), n_variables)]
  return simplify_cnf_formula(np.unique(np.vstack(new_ones), axis=0), n_variables)

def cnf_negation(matrix:np.array, n_variables:int) -> np.array:
  final = np.zeros((1, n_variables))
  clauses = -matrix
  while (clauses.shape[0] > 0):
    new_ones = []
    for ind_clause2 in range(n_variables):
      if (clauses[0,ind_clause2] != 0):
        temp = create_array(clauses[0,ind_clause2]*(ind_clause2+1), n_variables)
        new_ones += [delete_zeros((final+temp-temp*(final==temp))*((final*temp < 0).sum(axis=1) == 0).reshape(-1,1), n_variables)]
    final = np.unique(np.vstack(new_ones), axis=0)
    clauses = clauses[1:]
  return simplify_cnf_formula(final, n_variables)
def describe_network(bnn):
   for b in range(bnn.num_internal_blocks):
      for layer in bnn.blocks.layers:
         print('Input :',type(layer), layer.get_weights())
         print('Output :', layer.output)

def encode_network(the_model, input_file="BNN_CNFf.cnf") -> str:
    beginning = time.monotonic()
    n_inputs = the_model.blocks.layers[1].input_shape[1]
    inputs = [create_array(i, n_inputs) for i in range(1, n_inputs + 1)]
    n_layer = 1  # counter for tracking the current layer being processed
    print('Num layers:', len(the_model.blocks.layers))
    
    # Removed the first loop over blocks
    for layer_idx, layer in enumerate(the_model.blocks.layers):
        print(f'Layer {layer_idx}: {layer.__class__.__name__}')
        the_weights = layer.get_weights()
        outputs = []
        for id_neuron in range(layer.num_neurons):
            print('Num neuron in layer', layer.num_neurons)
            print(f'{seconds_separator(time.monotonic() - beginning)}   Layer: {len(the_model.blocks.layers) - 1}/{n_layer} | Neuron: {id_neuron + 1}/{layer.num_neurons}')
            print('the_weights[0] ', type(layer), layer.get_weights()[0])
            print(f"Shape of the_weights[0]: {the_weights[0].shape}")
            print(f"Value of id_neuron: {id_neuron}")
            print('BIAS :', (-the_weights[1][id_neuron]))
            weights_col = the_weights[0][:, id_neuron]
            print('SUM OF WEIGHTS :', weights_col.sum())
            print('LAST TERM : ', (the_weights[0][:, id_neuron] == -1).sum())
            D = ceil((-the_weights[1][id_neuron] + the_weights[0][:, id_neuron].sum()) / 2) + (weights_col == -1).sum()
            print('D Values:', D)
            previous = {}
            for id_input in range(len(inputs)):
                if (layer == the_model.blocks.layers[-1]): print(f'{seconds_separator(time.monotonic() - beginning)}   Working with the first {id_input + 1} inputs')
                actual = {}
                if (the_weights[0][id_input, id_neuron] == 1):
                    x = inputs[id_input]
                else:
                    x = cnf_negation(inputs[id_input], len(inputs))
                for d in range(D):
                    if (id_input < d): break
                    if (len(inputs) < id_input + 1 + D - (d + 1)): continue
                    if (d == 0):
                        if (id_input == 0):
                            actual[d] = x
                        else:
                            actual[d] = disjunction_cnfs(x, previous[d], len(inputs))
                    elif (id_input == d):
                        actual[d] = conjunction_cnfs(x, previous[d - 1], len(inputs))
                    else:
                        temp = conjunction_cnfs(x, previous[d - 1], len(inputs))
                        actual[d] = disjunction_cnfs(temp, previous[d], len(inputs))
                previous = actual
            outputs += [previous[D - 1].astype(dtype=np.int8)]
        inputs = outputs
        n_layer += 1

    print(f'Total time taken: {seconds_separator(time.monotonic() - beginning)}')
    dimacs_cnf = inputs[-1]
    dimacs_cnf = str(dimacs_cnf * np.arange(1, dimacs_cnf.shape[1] + 1)).replace(" 0", "").replace("]", "").replace("[", "").replace("\n", " 0\n") + " 0\n"
    while "  " in dimacs_cnf:
        dimacs_cnf = dimacs_cnf.replace("  ", " ")
    output_file = "output_final.cnf"
    with open(output_file, 'w') as f:
        f.write('p cnf %d %d\n' % (n_inputs, dimacs_cnf.count('\n')))
        f.write(dimacs_cnf)

    return output_file


def check_satisfiability(cnf_file_path,solver):
    

    # Load the CNF formula from the file
    with open(cnf_file_path, 'r') as file:
        cnf_formula = file.read()
        #print('cnf_formula :',cnf_formula)

    # Parse and add the CNF formula to the solver
    clauses = [list(map(int, line.split()[:-1])) for line in cnf_formula.splitlines() if line and line[0] not in ('c', 'p')]

    #print('Clauses :',clauses)
    for clause in clauses:
        #print(clause)
        solver.add_clause(clause)

    # Check satisfiability
    result = solver.solve()

    return result

