# Preparing the term1:
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
from itertools import chain
import time
import re
import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)
from scipy.optimize import minimize
optimizer = qml.AdamOptimizer(stepsize=0.5)
ash_excitation = []
energies = []
excitations= []
old_grad = []
operator_check = []  # To keep track of the highest operator values
alpha = qml.numpy.array(0.6 + 0.8j, requires_grad=True)  #Parameter for the excitation
beta = qml.numpy.array(0.6 + 0.8j, requires_grad=True)  #Parameter for the excitation
print('This is case 2 using Adapt like Gradients - s2 state is prepared with the latest excitation- Repetitive of operators')
X = qml.PauliX
Y = qml.PauliY
Z = qml.PauliZ
I = qml.Identity






def ags_exact(symbols, coordinates, active_electrons, active_orbitals, adapt_it, shots = None):
    print('Using active space, check if you change the H accordingly')
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, basis="sto-6g", method="pyscf",active_electrons=active_electrons, active_orbitals=active_orbitals)
    #print(H)
    hf_state = qchem.hf_state(active_electrons, qubits)
    #Calculation of HF state
    dev = qml.device("lightning.qubit", wires=qubits)
    @qml.qnode(dev)
    def circuit(hf_state, active_electrons, qubits, H): 
        qml.BasisState(hf_state, wires=range(qubits))
        return qml.expval(H)   #Calculating the expectation value of the Hamiltonian
    
    # Commutator calculation for HF state
    @qml.qnode(dev)
    def commutator_0(H,w, k):  #H is the Hamiltonian, w is the operator, k is the basis state - HF state
        qml.BasisState(k, wires=range(qubits))
        res = qml.commutator(H, w)   #Calculating the commutator
        return qml.expval(res)
    
    # Commutator calculation for other states except HF state
    @qml.qnode(dev)
    def commutator_1(H,w, k): #H is the Hamiltonian, w is the operator, k is the basis state
        qml.StatePrep(k, wires=range(qubits))
        res = qml.commutator(H, w) #Calculating the commutator
        return qml.expval(res)

    #Energy calculation 
    @qml.qnode(dev)
    def ash(params, ash_excitation, hf_state, H):
        #print('HF stat:', hf_state)
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]]  #Appln of HF state
        for i, excitation in enumerate(ash_excitation):
            if len(ash_excitation[i]) == 4:
                #print('The ash excitation now is', ash_excitation[i])
                #print('Wires1 Invert removed are ', list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)) )
                #print('Wires2 Invert removed are',list(range(ash_excitation[i][2], ash_excitation[i][3] + 1)))
                qml.FermionicDoubleExcitation(weight=params[i], wires1=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)), wires2=list(range(ash_excitation[i][2], ash_excitation[i][3] + 1)))
            elif len(ash_excitation[i]) == 2:
                qml.FermionicSingleExcitation(weight=params[i], wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)))
        return qml.expval(H)  #Calculating the expectation value of the Hamiltonian
    
    # Calculation of New state, same as the above function but with the state return
    dev1 = qml.device("lightning.qubit", wires=qubits)
    @qml.qnode(dev1)
    def new_state(hf_state, ash_excitation, params):
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]] #Applying the HF state
        for i, excitations in enumerate(ash_excitation):
            if len(ash_excitation[i]) == 4:
                qml.FermionicDoubleExcitation(weight=params[i], wires1=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)), wires2=list(range(ash_excitation[i][2], ash_excitation[i][3] + 1)))
            elif len(ash_excitation[i]) == 2:
                qml.FermionicSingleExcitation(weight=params[i], wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)))
        return qml.state()
    

    
    dev1 = qml.device("lightning.qubit", wires=qubits)
    @qml.qnode(dev1)
    def s1state(hf_state, ashs, params):
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]] #Applying the HF state
        #print('Params in s1state are', params)
        #print('Ashs in s1state are', ashs)
        for i, excitations in enumerate(ashs):
            if len(ashs[i]) == 4:
                qml.FermionicDoubleExcitation(weight=params[i], wires1=list(range(ashs[i][0], ashs[i][1] + 1)), wires2=list(range(ashs[i][2], ashs[i][3] + 1)))
            elif len(ashs[i]) == 2:
                qml.FermionicSingleExcitation(weight=params[i], wires=list(range(ashs[i][0], ashs[i][1] + 1)))
        return qml.state()


    


    @qml.qnode(dev1)
    def s2state(hf_state, ashs, params2):
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]] #Applying the HF state
        for i in range(len(latest_ash)):
            #print('The value of i', i)
            if len(latest_ash[i]) == 4:
                #print('The latest ash excitation in s2 state is', latest_ash[i])
                #print('params2 is', params2[i])
                qml.DoubleExcitation(params2[i], wires = latest_ash[i]) 
            elif len(latest_ash[i]) == 2:
                qml.SingleExcitation(params2[i], wires = latest_ash[i])
        return qml.state()
    dev2 = qml.device("lightning.qubit", wires=qubits, shots = 1000)
    @qml.qnode(dev2)
    def measure(state):
        qml.StatePrep(state, wires=range(qubits))
        return qml.counts()
    
    def global_cost(params, params2, alpha, beta):
        s1o = alpha * s1state(hf_state, ashs, params)
        s2o = beta * s2state(hf_state, latest_ash, params2)
        Ham_matrix = qml.matrix(H, wire_order=range(qubits))
        s1oc = qml.math.conj(s1o)
        s2oc = qml.math.conj(s2o)
        # Hamiltonian expectation calculations...
        term1 = qml.math.real(qml.math.dot(s1oc, qml.math.dot(Ham_matrix, s1o)))  # <S1|H|S1>
        #term2 = np.dot(s2oc, Ham_matrix @ s2o).real  # <S2|H|S2>
        term2 = qml.math.real(qml.math.dot(s2oc, qml.math.dot(Ham_matrix, s2o))) # <S2|H|S2>
        #term3 = np.dot(s1oc, Ham_matrix @ s2o).real  # <S1|H|S2>
        term3 = qml.math.real(qml.math.dot(s1oc, qml.math.dot(Ham_matrix, s2o)))  # <S1|H|S2>
        #term4 = np.dot(s2oc, Ham_matrix @ s1o).real  # <S2|H|S1>
        term4 = qml.math.real(qml.math.dot(s2oc, qml.math.dot(Ham_matrix, s1o)))  # <S2|H|S1>
        numerator = term1 + term2 + term3 + term4
        #D1 = np.dot(s1oc, s1o).real
        D1 = qml.math.real(qml.math.dot(s1oc, s1o))  # <S1|S1>
        #D2 = np.dot(s2oc, s2o).real
        D2 = qml.math.real(qml.math.dot(s2oc, s2o))  # <S2|S2>
        #D3 = np.dot(s1oc, s2o).real
        D3 = qml.math.real(qml.math.dot(s1oc, s2o))  # <S1|S2>
        #D4 = np.dot(s2oc, s1o).real
        D4 = qml.math.real(qml.math.dot(s2oc, s1o))  # <S2|H|S1>
        denominator = D1 + D2 + D3 + D4
        return numerator / denominator

    def global_cost_flat(x):
        params = x[:len(ashs)]  # 0,1
        params2 = x[len(ashs):len(ashs) + 1] #2
        alpha = x[len(ashs)+1] #3
        beta = x[len(ashs)+2]  #4
        return global_cost(params, params2, alpha, beta)

    #def callback(params):
        #print(f"Current parameters: {params}")
        #print(f"Current cost: {cost(params)}\n")
    

    print('HF state is', circuit(hf_state, active_electrons, qubits, H))
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)

    op1 =  [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "-"}) for x in singles]
    op2 =  [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "+", (2, x[2]): "-", (3, x[3]): "-"})for x in doubles]
    operator_pool = (op1) + (op2)  #Operator pool - Singles and Doubles
    print('Total excitations are', len(operator_pool))
    states = [hf_state]
    params = np.zeros(len(ash_excitation), requires_grad=True) 

    null_state = np.zeros(qubits,int)
    #print('Null state is', null_state)
    

    for j in range(adapt_it):
        print('The adapt iteration now is', j)  #Adapt iteration
        max_value = float('-inf')
        max_operator = None
        k = states[-1] if states else hf_state  # if states is empty, fall back to hf_state
       
        for i in operator_pool:
            w = qml.fermi.jordan_wigner(i)  #JW transformation
            if np.array_equal(k, hf_state): # If the current state is the HF state
                current_value = abs(2*(commutator_0(H, w, k)))      #Commutator calculation is activated  
            else:
                current_value = abs(2*(commutator_1(H, w, k)))      #For other states, commutator calculation is activated
            print(f'The expectation value of {i} is', current_value)

            if current_value > max_value:
                max_value = current_value
                max_operator = i

        print(f"The highest operator value is {max_value} for operator {max_operator}")  #Highest operator value

        old_grad.append(max_value)  #Appending the old gradient value
        operator_check.append(max_operator)
        indices_str = re.findall(r'\d+', str(max_operator))
        excitations = [int(index) for index in indices_str]
        print('Highest gradient excitation is', excitations)
        ash_excitation.append(excitations) #Appending the excitations to the ash_excitation
        ashs = ash_excitation
        print('Len of ashs:', len(ashs))
        alpha = qml.numpy.array(0.6 + 0.8j, requires_grad=True)
        print('Exc. going to prepare state1',ashs)
        params = np.zeros(len(ashs), requires_grad=True)    
        #params = np.append(params, 0.0)  #Parameters initialization



        s1ostate = alpha * s1state(hf_state, ashs, params)
        print(qml.draw(s1state, max_length=100)(hf_state,ashs,params))

        #s2 state preparation
        latest_ash = [ashs[-1]]
        print("Latest excitation:", latest_ash)
        print('Length of latest excitation:', len(latest_ash))
        params2 = qml.numpy.array([np.pi/4], requires_grad=True)  # Initial parameters for the excitations
        beta = qml.numpy.array(0.6 + 0.8j, requires_grad=True)
        print('Params2 are', params2)

        
        s2ostate = beta * s2state(hf_state, latest_ash, params2)
        print(qml.draw(s2state, max_length=100)(hf_state,latest_ash,params2))
        x0 = np.concatenate([params, params2, np.array([alpha]), np.array([beta])])

        print(x0)




        result = minimize(global_cost_flat, x0, method='BFGS')
        print(result)
        energies.append(result.fun)  # Appending the energy to the energies list
        params = result.x[:len(ashs)]  # Extracting the first part of the result as params
        params2 = result.x[len(ashs):len(ashs) + 1]  # Extracting the second part of the result as params2
        print('Params1 after optimization:', params)
        print('Params2 after optimization:', params2)
        alpha = result.x[len(ashs) + 1]  # Extracting the alpha parameter
        beta = result.x[len(ashs) + 2]  # Extracting the beta parameter
        print('Alpha after optimization:', alpha)
        print('Beta after optimization:', beta)

        #final_state = alpha * s1state(hf_state, ashs, params) + beta * s2state(hf_state, latest_ash, params2)
        #final_state = final_state / np.linalg.norm(final_state)

        final_state = s1state(hf_state, ashs, params)
        print("Norm after manual normalization:", np.linalg.norm(final_state)) 
        print('counts after the final state with optimization is', measure(final_state))
        states.append(final_state)  # Appending the new state to the states list
        
    return params, ash_excitation, qubits, H, energies, old_grad




symbols  = [ 'H', 'H', 'H', 'H']
print('H4-3A-GS-BFGS-sto6g')
r_bohr = 1.8897259886 
coordinates = np.array([[0.0,0.0, 0.0], [0.0, 0.0, 3.0*r_bohr], [0.0,0.0,6.0*r_bohr],[0.0, 0.0, 9.0*r_bohr]])



#electrons = 10  # 7 from N and 3 from H
#orbitals = 16
charge = 0

active_electrons = 4
active_orbitals = 4 #Thinkng it is spatial 


params, ash_excitation, qubits, H,energies, old_grad = ags_exact(symbols, coordinates, active_electrons, active_orbitals, shots = None, adapt_it=25) #1 is used for params



print('The params after GS is',params)
print('Ash excitation after gs state:', ash_excitation)
print('Energies:', energies)
print('Old gradient:', old_grad)    
