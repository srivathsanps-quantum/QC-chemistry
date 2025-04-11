import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
from itertools import chain
import time
import re
from scipy.optimize import minimize
ash_excitation = []
energies = []
excitations= []

X = qml.PauliX
Y = qml.PauliY
Z = qml.PauliZ
I = qml.Identity






def ags_exact(symbols, coordinates, active_electrons, active_orbitals, adapt_it, shots = None):
    print('Using active space, check if you change the H accordingly')
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, basis="sto-3g", method="pyscf",active_electrons=active_electrons, active_orbitals=active_orbitals)
    print(H)
    hf_state = qchem.hf_state(active_electrons, qubits)
    #Calculation of HF state
    dev = qml.device("lightning.qubit", wires=qubits)
    @qml.qnode(dev)
    def circuit(hf_state, active_electrons, qubits, H):
        #print('Updated hf_state is', hf_state)  
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
        [qml.PauliX(i) for i in np.nonzero(hf_state)[0]]  #Appln of HF state
        for i, excitation in enumerate(ash_excitation):
            if len(ash_excitation[i]) == 4:
                qml.FermionicDoubleExcitation(weight=params[i], wires1=ash_excitation[i][2:][::-1], wires2=ash_excitation[i][:2][::-1])
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
                qml.FermionicDoubleExcitation(weight=params[i], wires1=ash_excitation[i][2:][::-1], wires2=ash_excitation[i][:2][::-1])
            elif len(ash_excitation[i]) == 2:
                qml.FermionicSingleExcitation(weight=params[i], wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)))
        return qml.state()
    

    
    
    def cost(params):
        energy = ash(params, ash_excitation, hf_state, H)
        return energy

    def callback(params):
        print(f"Current parameters: {params}")
        print(f"Current cost: {cost(params)}\n")
    

    print('HF state is', circuit(hf_state, active_electrons, qubits, H))
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    op1 =  [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "-"}) for x in singles]
    op2 =  [qml.fermi.FermiWord({(0, x[0]): "+", (1, x[1]): "+", (2, x[2]): "-", (3, x[3]): "-"})for x in doubles]
    operator_pool = (op1) + (op2)  #Operator pool - Singles and Doubles
    print('Total excitations are', len(operator_pool))
    states = [hf_state]
    params = np.zeros(len(ash_excitation), requires_grad=True) 

    null_state = np.zeros(qubits,int)
    print('Null state is', null_state)

    for j in range(adapt_it):
        print('The adapt iteration now is', j)  #Adapt iteration
        max_value = float('-inf')
        max_operator = None
        k = states[-1] if states else hf_state  # if states is empty, fall back to hf_state
       
        for i in operator_pool:
            #print('The current excitation operator is', i)   #Current excitation operator - fermionic one
            w = qml.fermi.jordan_wigner(i)  #JW transformation
            if np.array_equal(k, hf_state): # If the current state is the HF state
                current_value = abs(2*(commutator_0(H, w, k)))      #Commutator calculation is activated  
            else:
                current_value = abs(2*(commutator_1(H, w, k)))      #For other states, commutator calculation is activated

            if current_value > max_value:
                max_value = current_value
                max_operator = i

        #print(f"The highest operator value is {max_value} for operator {max_operator}")  #Highest operator value


        indices_str = re.findall(r'\d+', str(max_operator))
        excitations = [int(index) for index in indices_str]
        print('Highest gradient excitation is', excitations)
        ash_excitation.append(excitations) #Appending the excitations to the ash_excitation

        params = np.append(params, 0.0)  #Parameters initialization



        #Energy calculation
        result = minimize(cost, params, method='powell', callback=callback, tol = 1e-12, options = {'disp': False, 'maxiter': 1e8})

        print("Final updated parameters:", result.x)
        print("Final cost:", result.fun)

        params= (result.x)
        energies.append(result.fun)


        ostate = new_state(hf_state, ash_excitation, params)
        #print(qml.draw(new_state, max_length=100)(hf_state,ash_excitation,params))
        gs_state = ostate
        states.append(ostate)
        
    return gs_state, params, ash_excitation, qubits, H

## So if you want the state, return the ostate and not states


import os
from time import time
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import scipy



def inite(elec,orb):
    config=[]
    list1=[]
    #singles
    for x in range(elec):
        count=orb-elec
        while (count<orb):
            for e in range(elec):
                if x==e:
                    if x%2==0:
                        config.append(count)
                        count=count+2
                    else:
                        config.append(count+1)
                        count=count+2
                else:
                    config.append(e)
                
            list1.append(config)
            config=[]
    #doubles
    for x in range(elec):
        for y in range(x+1,elec):
            count1=orb-elec
            count2=orb-elec
            for count1 in range(elec, orb, 2):
                for count2 in range(elec, orb, 2):
                    cont=0
                    if count1==count2:
                        if (x%2)!=(y%2):
                            cont=1
                    else:
                        cont=1
                    if (x%2)==(y%2) and count2<count1:
                        cont=0
                    if cont==1:    
                        for e in range(elec):
                            if x==e:
                                if x%2==0:
                                    config.append(count1)
                                else:
                                    config.append(count1+1)
                            elif y==e:
                                if y%2==0:
                                    config.append(count2)
                                else:
                                    config.append(count2+1)
                            else:
                                config.append(e)

                        list1.append(config)
                        config=[]
    return list1

def ee_exact(symbols, coordinates, active_electrons, active_orbitals ,params,ash_excitation, shots=1000):

    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, basis="sto-3g", method='pyscf', active_electrons=active_electrons, active_orbitals=active_orbitals)
    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    print('HF state:', hf_state)
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    wires=range(qubits)
    

    null_state = np.zeros(qubits,int)
    print('Null state is', null_state)
    list1 = inite(active_electrons,qubits)
    print('The list1 :', list1)
    values =[]
    for t in range(1):
        if shots==0:
            dev = qml.device("lightning.qubit", wires=qubits)
        else:

            dev = qml.device("lightning.qubit", wires=qubits,shots=shots)
        #circuit for diagonal part
        @qml.qnode(dev)
        def circuit_d(params, occ,wires, hf_state, ash_excitation):
            print('What is going  as hf_State:', hf_state)
            qml.BasisState(hf_state, wires=range(qubits))
            for w in occ:
                qml.X(wires=w)
            #Going to include excitations here
            for i, excitations in enumerate(ash_excitation):
                if len(ash_excitation[i]) == 4:
                    print('Exc. zstate:', ash_excitation[i])
                    print('Params in zstate:', params[i])
                    qml.FermionicDoubleExcitation(weight=params[i], wires1=ash_excitation[i][2:][::-1], wires2=ash_excitation[i][:2][::-1])
                elif len(ash_excitation[i]) == 2:
                    print('Single Exc. zstate:', ash_excitation[i])
                    print('Single params in zstate:', params[i])
                    qml.FermionicSingleExcitation(weight=params[i], wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)))
            return qml.expval(H)
        #circuit for off-diagonal part
        @qml.qnode(dev)
        def circuit_od(params, occ1, occ2,wires, hf_state, ash_excitation):
            print('What is going  as hf_State:', hf_state)
            qml.BasisState(hf_state, wires=range(qubits))
            for w in occ1:
                qml.X(wires=w)
            first=-1
            for v in occ2:
                if v not in occ1:
                    if first==-1:
                        first=v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first,v])
            for v in occ1:
                if v not in occ2:
                    if first==-1:
                        first=v
                        qml.Hadamard(wires=v)
                    else:
                        qml.CNOT(wires=[first,v])
            for i, excitations in enumerate(ash_excitation):
                if len(ash_excitation[i]) == 4:
                    print('Exc. zstate:', ash_excitation[i])
                    print('Params in zstate:', params[i])
                    qml.FermionicDoubleExcitation(weight=params[i], wires1=ash_excitation[i][2:][::-1], wires2=ash_excitation[i][:2][::-1])
                elif len(ash_excitation[i]) == 2:
                    print('Single Exc. zstate:', ash_excitation[i])
                    print('Single params in zstate:', params[i])
                    qml.FermionicSingleExcitation(weight=params[i], wires=list(range(ash_excitation[i][0], ash_excitation[i][1] + 1)))
            return qml.expval(H)
        #final M matrix
        M = np.zeros((len(list1),len(list1)))
        for i in range(len(list1)):
            for j in range(len(list1)):
                if i == j:
                    M[i,i] = circuit_d(params, list1[i], wires, null_state, ash_excitation)
        print("diagonal parts done")
        for i in range(len(list1)):
            for j in range(len(list1)):
                if i!=j:
                    Mtmp = circuit_od(params, list1[i],list1[j],wires, null_state, ash_excitation)
                    M[i,j]=Mtmp-M[i,i]/2.0-M[j,j]/2.0
        print("off diagonal terms done")
        #ERROR:not subtracting the gs energy
        eig,evec=np.linalg.eig(M)
        values.append(np.sort(eig))
    return values

symbols  = [ 'N', 'H', 'H', 'H']
print('NH3-1.0A-GS-Measurement shots + 1000 shots for Excited states')
r_bohr = 1.8897259886 
coordinates = np.array([[0.0,0.0, 0.0], [0.0, 0.0, 1.0*r_bohr], [0.950353*r_bohr,0.0,-0.336000*r_bohr],[-0.475176*r_bohr, -0.823029*r_bohr, -0.336000*r_bohr]])



electrons = 10  # 7 from N and 3 from H
orbitals = 20
charge = 0

active_electrons = 6
active_orbitals = 6


gs_state, params, ash_excitation, qubits, H = ags_exact(symbols, coordinates, active_electrons, active_orbitals, shots = None, adapt_it=15) #1 is used for params



print('The params after GS is',params)
print('Ash excitation after gs state:', ash_excitation)



eig = ee_exact(symbols, coordinates, active_orbitals,active_orbitals,params, ash_excitation)
print('exact eigenvalues:\n', eig)

dev2 = qml.device("lightning.qubit", wires=qubits, shots=10000)
@qml.qnode(dev2)
def measurement_shots(gs_state):
    qml.StatePrep(gs_state, wires= range(qubits))
    return qml.expval(H)

NoiseafterGS = measurement_shots(gs_state)
print('Noisy ground state:', NoiseafterGS)