import itertools
from math import factorial
import pickle
import numpy as np
from math import ceil

class InvalidInputs4dDBCSFi_2(Exception):
    def __init__(self):
        super().__init__("You need to pass a SDD or a binary perceptron.")

class InvalidInputs4SHAP(Exception):
    def __init__(self):
        super().__init__("You need to pass vectors with the correct number of variables.")

class dDBCSFi_2:
  def __init__(self, n_variables, SDD=None, perceptron=None):
    if (SDD == None) and (perceptron == None): raise InvalidInputs4dDBCSFi_2()
    self.SDD = SDD
    self.perceptron = perceptron
    self.formula = []
    self.variables = []
    self.n_vars = n_variables
    self.vars = np.arange(1, n_variables+1, dtype=np.int8)
    self.probabilities = [0.5]*n_variables

  def __compile_bnn_step(self, disjunction):
    formula = []
    set_o = set()
    #Iterates over all elements of the disjunction
    for conjunction in disjunction.elements():
      set_a = set()
      elements = []
      #Iterate over all the elements of the conjunction
      for element in conjunction:
        if (element.is_literal()): #Element
          ide = str(element)[13:str(element).index(",")]
          if (ide[:1] == "-"):
            set_a.add(self.vars[int(ide[1:])-1])
            ide = -self.vars[int(ide[1:])-1]
          else:
            set_a.add(self.vars[int(ide)-1])
            ide = self.vars[int(ide)-1]
        elif (element.is_decision()): #Formula
          ide = self.__compile_bnn_step(element)
          for id in ide[1]: set_a.add(self.vars[id-1])
          ide = ide[0]
          if (ide == []): ide = "0"
        elif (element.is_true()): ide = "1" #True
        else: ide = "0" #False
        if (ide != "1"):
          elements += [ide]
          if (ide == "0"): break
      #Check that the conjunction is not always F
      if ("0" in elements): continue
      else:
        #Check if the conjunction is always T
        if (elements == []):
          formula = "1"
          set_o = set()
          break
        else: #The conjunction varies
          if (not isinstance(formula, list)) or (formula != []): #(When the formula is a negative number, it seems to be confusing
                                                                 #it with a list, so I had to add a condition to see if it's a list)
            for id in set_o: #Check that the new formula matches the old
              if (id not in set_a):
                if (len(elements) == 1): elements = [False, elements[0], [True, self.vars[id-1], -self.vars[id-1]]]
                elif (len(elements) == 2): elements = [False, [False]+elements, [True, self.vars[id-1], -self.vars[id-1]]]
                else: elements = [False, elements, [True, self.vars[id-1], -self.vars[id-1]]]
            for id in set_a: #Check that the old formula matches the new
              if (id not in set_o):
                if (formula == []): formula = [True, self.vars[id-1], -self.vars[id-1]]
                else: formula = [False, formula, [True, self.vars[id-1], -self.vars[id-1]]]
                set_o.add(self.vars[id-1])
            #Combine the formulas
            if (len(elements) == 1): formula = [True, formula, elements[0]]
            elif (len(elements) == 2): formula = [True, formula, [False]+elements]
            else: formula = [True, formula, elements]
          else: #No need to smooth
            for id in set_a: set_o.add(self.vars[id-1])
            if (len(elements) == 1): formula = elements[0]
            else: formula = [False]+elements
    return formula, set_o

  def __print_formula_step(self, formula, depth=0):
    if (formula[0]): print(" "*depth+"Or")
    else: print(" "*depth+"And")
    if isinstance(formula[1], list):
      self.__print_formula_step(formula[1], depth+1)
    else:
      print(" "*(depth+1)+str(formula[1]))
    if isinstance(formula[2], list):
      self.__print_formula_step(formula[2], depth+1)
    else:
      print(" "*(depth+1)+str(formula[2]))

  def __evaluate_formula_step(self, formula, assignment=[]):
    #Check the first element
    if isinstance(formula[1], list): value1 = self.__evaluate_formula_step(formula[1], assignment)
    else: value1 = (assignment[abs(formula[1])-1] == np.sign(formula[1]))
    #Check the second element
    if isinstance(formula[2], list): value2 = self.__evaluate_formula_step(formula[2], assignment)
    else: value2 = (assignment[abs(formula[2])-1] == np.sign(formula[2]))
    #Evaluate the formula (I think they can be combined, but for now I don't need to and it's easier to read that way)
    if (formula[0]): #It's a disjunction
      if (value1) or (value2): finalvalue = True
      else: finalvalue = False
    else: #It's a conjunction
      if (value1) and (value2): finalvalue = True
      else: finalvalue = False
    return finalvalue

  def __count_nodes_step(self, formula):
    #Check the first element
    if isinstance(formula[1], list): value1 = self.__count_nodes_step(formula[1])
    else: value1 = 1
    #Check the second element
    if isinstance(formula[2], list): value2 = self.__count_nodes_step(formula[2])
    else: value2 = 1
    return value1+value2+1
  def __encode_perceptron(self):
    the_weights = self.perceptron.layers[-1].get_weights()[0].reshape(-1)
    the_bias = self.perceptron.layers[-1].get_weights()[1].reshape(-1)
    D = ceil((-the_bias + the_weights.sum())/2) + (the_weights == -1).sum()
    previous = {}
    smoothness = []
    for id_input in range(self.n_vars):
      actual = {}
      if (the_weights[id_input] == 1): x = self.vars[id_input]
      else: x = -self.vars[id_input]
      for d in range(D):
        if (id_input < d): break
        if (self.n_vars < id_input+1+D-(d+1)): continue
        if (d == 0):
          if (id_input == 0): actual[d] = x
          else: actual[d] = [True, [False, x, smoothness], [False, -x, previous[d]]]
        elif (id_input == d): actual[d] = [False, x, previous[d-1]]
        else: actual[d] = [True, [False, x, previous[d-1]], [False, -x, previous[d]]]
      previous = actual
      if (smoothness == []): smoothness = [True, self.vars[id_input], -self.vars[id_input]]
      else: smoothness = [False, smoothness, [True, self.vars[id_input], -self.vars[id_input]]]
    return previous[D-1], set(self.vars.flatten())




  def compile_bnn(self):
    if (self.SDD == None):
      print('Hier !!!')
      self.formula, self.variables = self.__encode_perceptron()
    else:
      if (self.SDD.is_false()): self.formula, self.variables = "1", set()
      elif (self.SDD.is_true()): self.formula, self.variables = [], set()
      elif (self.SDD.is_decision()):
        self.formula, self.variables = self.__compile_bnn_step(self.SDD)
        if (not isinstance(self.formula, list)) and (self.formula != "1"): self.formula = [self.formula]
      else:
        ide = str(self.SDD)[13:str(self.SDD).index(",")]
        if (ide[:1] == "-"): self.formula, self.variables = [self.vars[int(ide[1:])-1]], set([self.vars[int(ide[1:])-1]])
        else: self.formula, self.variables = [-self.vars[int(ide)-1]], set([self.vars[int(ide)-1]])

  def print_formula(self, profundidad=0):
    if (self.formula == "1"): print("It's always True")
    elif (self.formula == []): print("It's always False")
    elif (len(self.formula) != 1): self.__print_formula_step(self.formula, profundidad)
    else: print(self.formula[0])

  def evaluate_formula(self, assignment=[]):
    if (self.formula == "1"): return 1.0
    elif (self.formula == []): return 0.0
    elif (len(self.formula) != 1): return float(self.__evaluate_formula_step(self.formula, assignment))
    else: return float(assignment[abs(self.formula[0])-1] == np.sign(self.formula[0]))

  def evaluate_formulas(self, assignments=[]):
    results = []
    for assignment in np.array(assignments):
      results += [self.evaluate_formula(assignment)]
    return results

  def count_nodes(self):
    if (self.formula == "1") or (self.formula == []): return 1
    elif (len(self.formula) == 1): return 1
    else: return self.__count_nodes_step(self.formula)

  def change_probabilities(self, new_probabilities):
    self.probabilities = new_probabilities

  def __get_gammas_and_deltas(self, formula, n_variable, vector):
    #Check the first element
    if isinstance(formula[1], list): gammas1, deltas1, variables1 = self.__get_gammas_and_deltas(formula[1], n_variable, vector)
    else:
      if (formula[1] == n_variable):
        gammas1, deltas1 = [1], [0]
        variables1 = set([formula[1]])
      elif (formula[1] == -n_variable):
        gammas1, deltas1 = [0], [1]
        variables1 = set([abs(formula[1])])
      elif (formula[1] == abs(formula[1])):
        gammas1 = [self.probabilities[formula[1]-1], vector[formula[1]-1]]
        deltas1 = [self.probabilities[formula[1]-1], vector[formula[1]-1]]
        variables1 = set([formula[1]])
      else:
        gammas1 = [1-self.probabilities[abs(formula[1])-1], 1-vector[abs(formula[1])-1]]
        deltas1 = [1-self.probabilities[abs(formula[1])-1], 1-vector[abs(formula[1])-1]]
        variables1 = set([abs(formula[1])])
    #Check the second element
    if isinstance(formula[2], list): gammas2, deltas2, variables2 = self.__get_gammas_and_deltas(formula[2], n_variable, vector)
    else:
      if (formula[2] == n_variable):
        gammas2, deltas2 = [1], [0]
        variables2 = set([formula[2]])
      elif (formula[2] == -n_variable):
        gammas2, deltas2 = [0], [1]
        variables2 = set([abs(formula[2])])
      elif (formula[2] == abs(formula[2])):
        gammas2 = [self.probabilities[formula[2]-1], vector[formula[2]-1]]
        deltas2 = [self.probabilities[formula[2]-1], vector[formula[2]-1]]
        variables2 = set([formula[2]])
      else:
        gammas2 = [1-self.probabilities[abs(formula[2])-1], 1-vector[abs(formula[2])-1]]
        deltas2 = [1-self.probabilities[abs(formula[2])-1], 1-vector[abs(formula[2])-1]]
        variables2 = set([abs(formula[2])])
    gammas, deltas, variables = [], [], set()
    #It's a disjunction
    if (formula[0]):
      for l in range(len(gammas1)):
        gammas += [gammas1[l] + gammas2[l]]
        deltas += [deltas1[l] + deltas2[l]]
      variables = variables1.copy()
    #It's a conjunction
    else:
      variables = variables1.union(variables2)
      for l in range(len(variables) - (n_variable in variables) + 1):
        gamma_temp = 0
        delta_temp = 0
        for l1 in range(len(gammas1)):
          for l2 in range(len(gammas2)):
            if (l1+l2 == l):
              gamma_temp += gammas1[l1]*gammas2[l2]
              delta_temp += deltas1[l1]*deltas2[l2]
        gammas += [gamma_temp]
        deltas += [delta_temp]
    return gammas, deltas, variables

  def obtain_SHAP(self, n_variable, vector):
    if (len(vector) != len(self.probabilities)): raise InvalidInputs4SHAP()
    if (self.formula == "1"): return 1/len(self.probabilities)
    elif (self.formula == []): return -1/len(self.probabilities)
    else:
      gammas, deltas = self.__get_gammas_and_deltas(self.formula, n_variable, vector)[:2]
      SHAP = 0
      n_X = len(gammas)
      for i in range(n_X):
        SHAP += (factorial(i)*factorial(n_X-i-1)/factorial(n_X))*(vector[n_variable-1] - self.probabilities[n_variable-1])*(gammas[i] - deltas[i])
      return SHAP

  def obtain_SHAPs(self, vector):
    SHAPs = []
    for i in range(len(vector)):
      SHAPs += [self.obtain_SHAP(i+1, vector)]
    return SHAPs

  def __form_graph(self, formula, level=0):
    self.the_level = min(self.the_level, level)
    if (formula[0]): #It's a disjunction
      id_a = "o"+str(self.ind_o)
      self.ind_o += 1
      self.G.add_node(id_a, layer=level)
      self.nodes_color += ["hotpink"]
      self.labels[id_a] = r"$\vee$"
    else: #It's a conjunction
      id_a = "y"+str(self.ind_a)
      self.ind_a += 1
      self.G.add_node(id_a, layer=level)
      self.nodes_color += ["c"]
      self.labels[id_a] = r"$\wedge$"
    for i in [1,2]:
      if (isinstance(formula[i], list)):
        id_b = self.__form_graph(formula[i], level-1)
      else:
        if (formula[i] == abs(formula[i])):
          id_b = "e"+str(self.ind_E)
          self.ind_E += 1
          self.G.add_node(id_b, layer=level-1)
          self.nodes_color += ["moccasin"]
          self.labels[id_b] = fr"$x_{formula[i]}$"
        else:
          id_b = "e"+str(self.ind_E)
          self.ind_E += 1
          self.G.add_node(id_b, layer=level-1)
          self.nodes_color += ["limegreen"]
          self.labels[id_b] = fr"$-x_{abs(formula[i])}$"
      self.G.add_edge(id_a, id_b)
    return id_a

  def __form_graph_alt(self, formula, level=0):
    self.the_level = min(self.the_level, level)
    if (formula[0]): #It's a disjunction
      id_a = "o"+str(self.ind_o)
      self.ind_o += 1
      self.G.add_node(id_a, layer=level)
      self.nodes_color += ["hotpink"]
      self.labels[id_a] = r"$\vee$"
    else: #It's a conjunction
      id_a = "y"+str(self.ind_a)
      self.ind_a += 1
      self.G.add_node(id_a, layer=level)
      self.nodes_color += ["c"]
      self.labels[id_a] = r"$\wedge$"
    for i in [1,2]:
      if (isinstance(formula[i], list)):
        id_b = self.__form_graph_alt(formula[i], level-1)
      else:
        if (formula[i] == abs(formula[i])):
          id_b = "e"+str(self.ind_E)
          self.ind_E += 1
          self.G.add_node(id_b, layer=level-1)
          self.nodes_color += ["moccasin"]
          self.labels[id_b] = fr"$x_{formula[i]}$"
        else:
          id_b = "n"+str(self.ind_n)
          id_c = "e"+str(self.ind_E)
          self.ind_n += 1
          self.ind_E += 1
          self.G.add_node(id_b, layer=level-1)
          self.G.add_node(id_c, layer=level-2)
          self.nodes_color += ["limegreen"]
          self.nodes_color += ["moccasin"]
          self.labels[id_b] = fr"$-$"
          self.labels[id_c] = fr"$x_{abs(formula[i])}$"
          self.G.add_edge(id_c, id_b)
      self.G.add_edge(id_b, id_a)
    return id_a

  def plot_formula(self, resplt=10, alternative=False):
    self.labels = {}
    self.nodes_color = []
    self.ind_a = 1
    self.ind_o = 1
    self.ind_T = 1
    self.ind_E = 1
    self.ind_n = 1
    self.the_level = 0
    self.G = nx.DiGraph()
    if (self.formula == "1"):
      self.G.add_node("T", layer=1)
      self.nodes_color += ["coral"]
      self.labels["T"] = fr"$T$"
    elif (self.formula == []):
      self.G.add_node("F", layer=1)
      self.nodes_color += ["thistle"]
      self.labels["F"] = fr"$F$"
    elif (len(self.formula) == 1):
      if (self.formula[0] == abs(self.formula[0])):
        self.G.add_node("e", layer=1)
        self.nodes_color += ["moccasin"]
        self.labels["e"] = fr"$x_{self.formula[0]}$"
      else:
        self.G.add_node("e", layer=1)
        self.nodes_color += ["limegreen"]
        self.labels["e"] = fr"$-x_{abs(self.formula[0])}$"
    elif (alternative): ignorable = self.__form_graph_alt(self.formula)
    else: ignorable = self.__form_graph(self.formula)
    if (self.the_level == 0): self.the_level = 1
    pos = nx.multipartite_layout(self.G, subset_key="layer", align="horizontal")
    plt.clf()
    plt.figure(figsize=[resplt,resplt])
    nx.draw_networkx(self.G, pos, node_size=200*resplt/abs(self.the_level), with_labels=True, node_color=self.nodes_color,
                     labels=self.labels, font_size=16*resplt/(abs(self.the_level)+resplt), arrows=alternative, width=(1+alternative))
    plt.show()
    del self.labels, self.nodes_color, self.ind_a, self.ind_o, self.ind_T, self.ind_E, self.the_level, self.G

  def corroborate_equivalence(self, the_model, test=-1):
    equivalent = True
    combinations = np.array(list(itertools.product([-1, 1], repeat=self.n_vars)))

    if test < 0:
        relevants = len(combinations)
    else:
        relevants = test

    for i in combinations[:relevants]:
        # Evaluate the neural network for the current input
        res_net = the_model.predict(i.reshape(1, -1), verbose=0)[0][0]

        # Evaluate the compiled formula for the same input
        res_cir = self.evaluate_formula(i)

        # Check if the results are not equal
        if res_net != res_cir:
            equivalent = False
            break

    # Print the number of nodes for the model and the formula
    model_nodes = the_model.count_params()  # Assuming count_params() returns the number of nodes in the model
    formula_nodes = self.count_nodes()

    print(f"Number of nodes in the model: {model_nodes}")
    print(f"Number of nodes in the formula: {formula_nodes}")

    if equivalent:
        print(f'They are equivalent for the {min(relevants, len(combinations))} values reviewed')
    else:
        print("They aren't equivalent")
