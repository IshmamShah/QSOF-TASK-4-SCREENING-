#!/usr/bin/env python
# coding: utf-8

# # QOSF Task 4 Solution

# SHAH ISHMAM MOHTASHIM, 
# DEPARTMENT OF CHEMISTRY, UNIVERSITY OF DHAKA
# 
# Email: sishmam51@gmail.com

# The given task 4 was
# Finding the lowest eigenvalue of the following matrix:
#  $$\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
# 
# 
# using VQE-like circuits, from scratch.
# 

# # Solution

# The given matrix has to be first decomposed into the sum of the Pauli terms.
# The four Pauli terms are:
# 

# # Decomposition of Matrix

# I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
# X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
# Y = \begin{pmatrix} 0 &-i \\ i & 0 \end{pmatrix}
# Z = \begin{pmatrix} 1 & 0 \\ 0 &-1 \end{pmatrix}

# The given matrix can be decomposed using
# 
#  $H = \sum_{i,j=1,x,y,z} a_{i,j} \left( \sigma_i \otimes \sigma_j \right),
# \quad$
#  $a_{i,j} = \frac{1}{4} tr \left[\left( \sigma_i \otimes \sigma_j \right) H \right]$
# 
# 
# Reference- https://michaelgoerz.net/notes/decomposing-two-qubit-hamiltonians-into-pauli-matrices.html

# The decomposition is implemented in code below:

# In[814]:


#importing required dependencies for circuit formation
from qiskit import *
import numpy as np 
from numpy import kron


# Reference for the docomposition function: https://michaelgoerz.net/notes/decomposing-two-qubit-hamiltonians-into-pauli-matrices.html

# In[815]:


def HS(M1, M2):
    """Hilbert-Schmidt-Product of two matrices M1, M2"""
    return (np.dot(M1.conjugate().transpose(), M2)).trace()



def decompose(H):
    """Decompose Hermitian 4x4 matrix H into Pauli matrices"""
    from numpy import kron
    sx = np.array([[0, 1],  [ 1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0],  [0, -1]], dtype=np.complex128)
    id = np.array([[1, 0],  [ 0, 1]], dtype=np.complex128)
    S = [id, sx, sy, sz]
    labels = ['I', 'sigma_x', 'sigma_y', 'sigma_z']
    for i in range(4):
        for j in range(4):
            label = labels[i] + labels[j]
            a_ij = 0.25 * HS(kron(S[i], S[j]), H)
            if a_ij != 0.0:
                print(a_ij,'*' ,label, '+')


# Constructing the given matrix M:

# In[816]:


M = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


# Decomposing we get

# In[817]:


decompose(M)


# # The decomposed matrix is 
# $(0.5+0j) * II +
# (-0.5+0j) * XX +
# (-0.5+0j) * YY +
# (0.5+0j) * ZZ$
# 
# 
# 
# 

# The decomposed matrix can be written as hamiltonian
# H1+H2+H3+H4

# # Construction of Ansatz

# The ansatz we use is: (RX I) CX (HI) |00>, where angle in RX is the variational parameter. (Provided by the hints from the task document)
# 

# In[818]:


def ansatz(angle):
    '''Circuit of two qubits'''
    ckt = QuantumCircuit(2) 
    ckt.h(0)
    ckt.cx(0, 1)
    ckt.rx(angle, 0)
  
    
    return ckt


# In[819]:


'''
Figure of Ansatz circuit
'''
ckt.draw(output='mpl')


# To meausre the expecatation values, appropriate rotations are to be applied to the circuit. 
# Reference: https://www.mustythoughts.com/variational-quantum-eigensolver-explained

# # Circuit Preparation for ZZ

# To meausre the expectation value for ZZ, rotations are not needed.
# 

# In[820]:


def zz_circuit(ckt):
    zz_measure = ckt.copy()
    zz_measure.measure_all()
    return zz_measure

zz_measure = zz_circuit(ckt)
zz_measure.draw(output='mpl')
'''
Figure of the circuit for ZZ
'''


# # Circuit Preparation for XX
# 

# For X, $RY(-π/2)$ rotation is needed

# In[821]:


def xx_circuit(ckt):
    xx_measure = ckt.copy()
    
    '''
    Rotation by RY(-pie/2) for both qubits as shown by the figure of the circuit
    '''
    
    xx_measure.barrier()
    xx_measure.ry(-np.pi/2,0)
    xx_measure.ry(-np.pi/2,1)
    xx_measure.measure_all()
    
    return xx_measure

xx_measure = xx_circuit(ckt)
xx_measure.draw(output='mpl')


# # Circuit Preparation for YY

# For Y, $RX(π/2)$ rotation is needed

# In[822]:


def yy_circuit(ckt):
    yy_measure = ckt.copy()
    '''
    Rotation by RX(pie/2) as shown by the figure of the circuit
    '''
    yy_measure.barrier()
    yy_measure.rx(np.pi/2, 0)
    yy_measure.rx(np.pi/2, 1)
    yy_measure.measure_all()
    
    return yy_measure

yy_measure = yy_circuit(ckt)
yy_measure.draw(output='mpl')


# # Measurement of Expectation Values

# To simulate the circuit for expectation values, qasm simulator is used. For measurement of the expectation value, each of the pauli terms XX, YY, ZZ are found out individually. The II term is added as a constant and no meausurement is necessary. 
# 
# We measure the states(ansatz) $| \psi \rangle$ 1000 times(shots) using the basis in accordance with the individual terms of the decomposed Pauli Matrix.(XX, YY, ZZ- The individual terms guides the ansatz in making the measurement at a particular basis).
# 
# The values of the measurements are found in values of either 1 or 0 for X and Y and for Z the values are found to be 1(for 0) and -1(for 1). Then the values are normalized over the number of shots(=1000). 
# 
# 

# In[823]:


simulator = Aer.get_backend('qasm_simulator')


# For ZZ meausrement:

# In[824]:


def ZZ(ckt, shots = 1024):
    
    zz_measure = zz_circuit(ckt)
    
    result = execute(zz_measure, backend = simulator, shots = shots).result()
    
    
    '''
    (number of  ( 00 )  and (11) states) - (the number of (01) and (10) state)
    normalized over the number of shots
    00 and 11 has positive signs after application of ZZ gates and 01 and 10 has negative signs
    '''
    
    
    items =result.get_counts(zz_measure).items()
    
    s = 0
    for key, counts in items:
        s+= (-1)**(int(key[0])+int(key[1]))*counts
    
    s = s/shots
        
    return s

    print(s)
    


# For XX measurement

# In[825]:


def XX(ckt, shots = 1024):
    
    xx_measure = xx_circuit(ckt)
    
    result = execute(xx_measure, backend = simulator, shots = shots).result()
    
    '''
    (number of  ( 00 )  and (11) states) - (the number of (01) and (10) state)
    normalized over the number of shots
    00 and 11 has positive signs after application of ZZ gates and 01 and 10 has negative signs
    '''
    
    
    items =result.get_counts(xx_measure).items()
    
    s = 0
    for key, counts in items:
        s+= (-1)**(int(key[0])+int(key[1]))*counts
    
    s = s/shots
        
    return s
    
    


# For YY measurement

# In[826]:


def YY(ckt, shots = 1024):
    
    yy_measure = yy_circuit(ckt)
    
    result = execute(yy_measure, backend = simulator, shots = shots).result()
    
    
    
    '''
    (number of  ( 00 )  and (11) states) - (the number of (01) and (10) state)
    normalized over the number of shots
    00 and 11 has positive signs after application of ZZ gates and 01 and 10 has negative signs
    '''
    
    items =result.get_counts(yy_measure).items()
    
    s = 0
    for key, counts in items:
        s+= (-1)**(int(key[0])+int(key[1]))*counts
    
    s = s/shots
        
    return s
    
  
   


# In[827]:


def H(ckt, shots = 1024):
    
    zz = ZZ (ckt, shots=1024)
    xx = XX (ckt, shots=1024)
    yy = YY (ckt, shots=1024)
    
    E = 0.5*1 + (-0.5)*xx + (-0.5)*yy + 0.5*zz 
    '''
    All the individual pauli terms/decomposed hamiltonions are added up 
    '''
 
    
    return E


# In[828]:


minimum_energy = 100
for i in range(0, 361):
    ckt = ansatz(i*np.pi/180)
    energy = H(ckt) 
    
    if (minimum_energy > energy):
        minimum_energy = energy

print("The lowest eigenvalue of the given matrix is",minimum_energy)


# Checking the minimum eigenvalue directly:

# In[829]:


w = np.linalg.eigvals(M)
print(f'Lowest eigenvalue found classicaly is {min(w)}')

