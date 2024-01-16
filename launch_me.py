from pysdd.sdd import SddManager
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from bnn import *
from bnn_to_cnf import *
from cnf_to_bdd import *
from bnn_to_sdd import *
from  pysat.solvers import Glucose3
import itertools
import numpy as np 
import pandas as pd
from tensorflow.keras import  optimizers

def read_and_print_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                print(line.strip())  # Strip to remove trailing newline characters
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
     bnn_model = BNN(num_dense_layer=3,num_neuron_in_dense_layer=200,num_neuron_output_layer=1)
     dimacs_file_path=encode_network(bnn_model)
     #describe_network(bnn_model)

     #dimacs_file_path = 'output_final.cnf'
     #n_vars = 10 
     #CNF to BDD
     #cnf_formula = read_dimacs_file(dimacs_file_path)
     #output_file_path = 'output_bdd_info.txt'

     #bdd_compiler = BDD_Compiler(n_vars, cnf_formula)
     #bdd = bdd_compiler.compile(output_file=output_file_path)
     #bdd.print_info(n_vars)
    

     # Replace 'your_data.txt' with the actual file path
     """
     file_path = 'data.txt'

    # Load data from the .txt file
     data = np.loadtxt(file_path, dtype=str)  # Assuming your data is in string format

    # Assuming each line in the file contains a long bit sequence followed by a 4-bit response
     X = data[:, 0]  # Features (long bit sequence)
     Y = data[:, 1]  # Labels (4-bit response)

     X = np.core.defchararray.replace(X, '?', '')
     data_dict = {'X': X, 'Y': Y}

    # Convert the dictionary to a pandas DataFrame
     df = pd.DataFrame(data_dict)

    # Display the DataFrame
     print(df.head())
     #df['X'] = label_encoder.fit_transform(df['X'])
     #df['Y'] = label_encoder.fit_transform(df['Y'])
     X = np.array(df['X'])
     Y = np.array(df['Y'])
     print('Features :',X)
     print('Y',Y)
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
     model = BNN(num_dense_layer=3, num_neuron_in_dense_layer=3, num_neuron_output_layer=1, input_shape=(17,17))
     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)


     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
     #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Convert binary strings to numerical arrays
     max_length = max(len(binary) for binary in X)

# Pad or truncate binary strings to the maximum length
     X_padded = pad_sequences([[int(bit) for bit in binary] for binary in X], maxlen=max_length, padding='post', dtype='float32')

     #X_padded = tf.convert_to_tensor([[int(bit) for bit in binary] for binary in X], dtype=tf.float32)
     Y_numeric = tf.keras.utils.to_categorical(Y)

    # Train the model
     model.fit(X_padded, Y_numeric, epochs=500)

    # Evaluate the model
     model.evaluate(X_padded, Y_numeric)
     """



     #BNN to SDD
     n_vars=10
     ddbcsfi_instance = dDBCSFi_2(n_variables=n_vars, perceptron=bnn_model)

     beginning = time.monotonic()
     #cnff_name = 'output_final.cnf'
     #cnff_name = encode_network(bnn_model, cnff_name)
     read_and_print_file(dimacs_file_path)
     duration = time.monotonic() - beginning
     print("Time taken to create the formula:", seconds_separator(duration),"\n")

     beginning = time.monotonic()
     mgr = SddManager()
     ssd_manager, node = mgr.from_cnf_file(bytes(dimacs_file_path, encoding='utf-8'))
     duration = time.monotonic() - beginning
     print("Time taken to create the SDD:", seconds_separator(duration),"\n")

     beginning = time.monotonic()
     The_circuit = dDBCSFi_2(10, SDD=node) # 10 is the number of inputs features or neurons
     The_circuit.compile_bnn()
     duration = time.monotonic() - beginning
     print("Time taken to create the dDBCSFi(2):", seconds_separator(duration))
     beginning = time.monotonic()
     The_circuit.corroborate_equivalence(bnn_model, -1)
     duration = time.monotonic() - beginning
     print("Time taken to verify the equivalence of dDBCSFi(2) with the BNN:", seconds_separator(duration))
     print("\nThe circuit has", The_circuit.count_nodes(), "nodes")

   
