import numpy as np
from inspect import isfunction
from qiskit import QuantumCircuit
from qiskit import transpile  
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info import hellinger_distance
from qiskit_ibm_runtime.fake_provider import FakeBurlingtonV2 as FakeDevice
from qiskit_aer import AerSimulator
from qiskit import transpile

### NO MODIFICAR ###

def test_1( qc_ghz_op : QuantumCircuit ):

    qc_ghz_op = transpile( qc_ghz_op )
    n_qubits = 4 
    qc_ghz = QuantumCircuit( n_qubits ) 
    qc_ghz.h(0)
    for j in range(n_qubits-1):
        qc_ghz.cx(j,j+1)
    state = Statevector( qc_ghz )
    state_op = Statevector( qc_ghz_op )

    if not isinstance( qc_ghz_op, QuantumCircuit ):
        print('No es un circuito cuántico')
    elif qc_ghz_op.num_qubits != n_qubits:
        print('El circuito no tiene 4 qubits')
    elif not np.isclose( np.linalg.norm( state-state_op ), 0 ):
        print('El circuito no prepara un estado GHZ')
    elif not ( qc_ghz_op.depth() == 3  ):
        print('La profundidad del circuito es muy grande')
    else:
        print('Felicidades, tu solución es correcta!')

def test_2( qc_ghz_op ):

    n_qubits = 4 
    qc_ghz_device = QuantumCircuit( n_qubits ) 
    qc_ghz_device.h(0)
    for j in range(n_qubits-2):
        qc_ghz_device.cx(j,j+1)

    qc_ghz_device.cx(2,1)
    qc_ghz_device.cx(1,2)
    qc_ghz_device.cx(2,1)
    qc_ghz_device.cx(1,3)

    device_backend = FakeDevice() 
    simulator_noise = AerSimulator.from_backend(device_backend) 
    qc_ghz_device_measured = qc_ghz_device.copy() 
    qc_ghz_device_measured.measure_all() 
    qc_ghz_device_measured =  transpile( qc_ghz_device_measured, 
                                        device_backend, optimization_level=0 ) 

    counts_ideal = { '0000':500, '1111':500  }

    qc_ghz_op_measured = qc_ghz_op.copy() 
    qc_ghz_op_measured.measure_all() 
    qc_ghz_op_measured =  transpile( qc_ghz_op_measured, device_backend, optimization_level=0 ) 

    state = Statevector( qc_ghz_device )
    state_op = Statevector( qc_ghz_op )

    if not isinstance( qc_ghz_op, QuantumCircuit ):
        print('No es un circuito cuántico')
    elif qc_ghz_op.num_qubits != n_qubits:
        print('El circuito no tiene 4 qubits')
    elif not np.isclose( np.linalg.norm( state-state_op ), 0 ):
        print('El circuito no prepara un estado GHZ')
    elif not ( qc_ghz_op_measured.depth() == 5  ):
        print('La profundidad del circuito mapeado al circuito es muy grande')
    else:
        error1 = 0
        error2 = 0
        for _ in range(100):
            counts_op = simulator_noise.run( qc_ghz_op_measured ).result().get_counts() 
            counts_device = simulator_noise.run( qc_ghz_device_measured ).result().get_counts() 

            error1 += hellinger_distance(counts_ideal, counts_device)
            error2 += hellinger_distance(counts_ideal, counts_op)

        if error2 < error1:
            print( 'Felicidades, tu solución es correcta!' )
        else:
            print( 'El error de tu circuito es mayor!')

def test_2a( Fourier ):

    sol = False 
    if not isfunction( Fourier ):
        print('Input no es una función')
    else:
        for num_qubits in range(2,6):

            F = np.exp( 2j*np.pi*np.outer( np.arange(2**num_qubits), np.arange(2**num_qubits) )/2**num_qubits  ) / np.sqrt(2**num_qubits)

            qc = Fourier( num_qubits )
            if not np.isclose( np.linalg.norm( F-Operator(qc).to_matrix() ), 0 ) :
                sol = False 
                print( 'La función no implementa la transformada de Fourier para {} qubits'.format(num_qubits) )
                break
            else:
                sol = True 

    if sol: 
        print('Felicidades, tu solución es correcta!')


def test_2b( U_to_n ):
    sol = False 
    for power in range(1,6):
        U1 = np.diag([1,1,1,np.exp(power*1j*2*np.pi*0.375)])
        U2 = Operator( U_to_n(power) ).to_matrix() 
        if not np.isclose( np.linalg.norm( U1-U2 ), 0 ) :
            sol = False 
            print( 'La función no implementa '+r'$U^n$'+' para potencia {}'.format(power) )
            break
        else:
            sol = True 

    if sol: 
        print('Felicidades, tu solución es correcta!')


def test_2c( QuantumPhaseEstimation ):

    sol = False
    for num_qubits in range(3,6):
        phi = 0.375 
        qc = QuantumPhaseEstimation(num_qubits)
        backend = AerSimulator()
        job = backend.run( qc )
        counts = job.result().get_counts()
        phi_hat = int( max(counts ), 2 ) / 2**num_qubits
        if not np.isclose( np.abs(phi_hat-phi), 0 ) :
            sol = False 
            print( 'La función no estima correctamente la fase para {} qubits'.format(num_qubits) )
            print( r'$\tilde\phi=$'+'{}'.format(phi_hat))
            break
        else:
            sol = True 

    if sol: 
        print('Felicidades, tu solución es correcta!')

####################################

def test_3a( Aj, Bk ):

       A = np.zeros((3,2,2), dtype=complex)
       B = np.zeros((3,2,2), dtype=complex)

       A[0] = np.array([[1.+0.j, 0.+0.j],
              [0.+0.j, 1.+0.j]])
       A[1] = np.array([[ 0.92387953+0.j,  0.38268343+0.j],
              [-0.38268343+0.j,  0.92387953+0.j]])
       A[2] = np.array([[ 0.70710678+0.j,  0.70710678+0.j],
              [-0.70710678+0.j,  0.70710678+0.j]])
       B[0] = np.array([[ 0.92387953+0.j,  0.38268343+0.j],
              [-0.38268343+0.j,  0.92387953+0.j]])
       B[1] = np.array([[ 0.70710678+0.j,  0.70710678+0.j],
              [-0.70710678+0.j,  0.70710678+0.j]])
       B[2] = np.array([[ 0.38268343+0.j,  0.92387953+0.j],
              [-0.92387953+0.j,  0.38268343+0.j]])
       
       sol = True
       for j, a in enumerate(Aj):
              if a.num_qubits > 1:
                     print('Los circuitos deben tener 1 qubit')
              if np.isclose( np.linalg.norm( Operator(a).to_matrix() - A[j] ), 0 ):
                     pass
              else:
                     sol = False 
                     print('El circuito {} de Alice no es correcto'.format(j))

       for k, b in enumerate(Bk):
              if b.num_qubits > 1:
                     print('Los circuitos deben tener 1 qubit')
              if np.isclose( np.linalg.norm( Operator(b).to_matrix() - B[k] ), 0 ):
                     pass
              else:
                     sol = False
                     print('El circuito {} de Bob no es correcto'.format(k) )

       if sol:
              print('Felicitaciones, tu solución es correcta!')

def test_3b( qcs ):
    Ops_2qb = np.array([[[ 2.70598050e-01+0.j,  2.70598050e-01+0.j,  6.53281482e-01+0.j,
          6.53281482e-01+0.j],
        [-6.53281482e-01+0.j,  6.53281482e-01+0.j, -2.70598050e-01+0.j,
          2.70598050e-01+0.j],
        [ 6.53281482e-01+0.j,  6.53281482e-01+0.j, -2.70598050e-01+0.j,
         -2.70598050e-01+0.j],
        [ 2.70598050e-01+0.j, -2.70598050e-01+0.j, -6.53281482e-01+0.j,
          6.53281482e-01+0.j]],

       [[ 5.00000000e-01+0.j,  5.00000000e-01+0.j,  5.00000000e-01+0.j,
          5.00000000e-01+0.j],
        [-5.00000000e-01+0.j,  5.00000000e-01+0.j, -5.00000000e-01+0.j,
          5.00000000e-01+0.j],
        [ 5.00000000e-01+0.j,  5.00000000e-01+0.j, -5.00000000e-01+0.j,
         -5.00000000e-01+0.j],
        [ 5.00000000e-01+0.j, -5.00000000e-01+0.j, -5.00000000e-01+0.j,
          5.00000000e-01+0.j]],

       [[ 6.53281482e-01+0.j,  6.53281482e-01+0.j,  2.70598050e-01+0.j,
          2.70598050e-01+0.j],
        [-2.70598050e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
          6.53281482e-01+0.j],
        [ 2.70598050e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
         -6.53281482e-01+0.j],
        [ 6.53281482e-01+0.j, -6.53281482e-01+0.j, -2.70598050e-01+0.j,
          2.70598050e-01+0.j]],

       [[ 2.77555756e-17+0.j,  5.00000000e-01+0.j,  5.00000000e-01+0.j,
          7.07106781e-01+0.j],
        [-7.07106781e-01+0.j,  5.00000000e-01+0.j, -5.00000000e-01+0.j,
          2.77555756e-17+0.j],
        [ 7.07106781e-01+0.j,  5.00000000e-01+0.j, -5.00000000e-01+0.j,
         -2.77555756e-17+0.j],
        [ 2.77555756e-17+0.j, -5.00000000e-01+0.j, -5.00000000e-01+0.j,
          7.07106781e-01+0.j]],

       [[ 2.70598050e-01+0.j,  6.53281482e-01+0.j,  2.70598050e-01+0.j,
          6.53281482e-01+0.j],
        [-6.53281482e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
          2.70598050e-01+0.j],
        [ 6.53281482e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
         -2.70598050e-01+0.j],
        [ 2.70598050e-01+0.j, -6.53281482e-01+0.j, -2.70598050e-01+0.j,
          6.53281482e-01+0.j]],

       [[ 5.00000000e-01+0.j,  7.07106781e-01+0.j,  5.55111512e-17+0.j,
          5.00000000e-01+0.j],
        [-5.00000000e-01+0.j,  5.55111512e-17+0.j, -7.07106781e-01+0.j,
          5.00000000e-01+0.j],
        [ 5.00000000e-01+0.j,  5.55111512e-17+0.j, -7.07106781e-01+0.j,
         -5.00000000e-01+0.j],
        [ 5.00000000e-01+0.j, -7.07106781e-01+0.j, -5.55111512e-17+0.j,
          5.00000000e-01+0.j]],

       [[-2.70598050e-01+0.j,  6.53281482e-01+0.j,  2.70598050e-01+0.j,
          6.53281482e-01+0.j],
        [-6.53281482e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
         -2.70598050e-01+0.j],
        [ 6.53281482e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
          2.70598050e-01+0.j],
        [-2.70598050e-01+0.j, -6.53281482e-01+0.j, -2.70598050e-01+0.j,
          6.53281482e-01+0.j]],

       [[ 0.00000000e+00+0.j,  7.07106781e-01+0.j,  0.00000000e+00+0.j,
          7.07106781e-01+0.j],
        [-7.07106781e-01+0.j,  0.00000000e+00+0.j, -7.07106781e-01+0.j,
          0.00000000e+00+0.j],
        [ 7.07106781e-01+0.j,  0.00000000e+00+0.j, -7.07106781e-01+0.j,
          0.00000000e+00+0.j],
        [ 0.00000000e+00+0.j, -7.07106781e-01+0.j,  0.00000000e+00+0.j,
          7.07106781e-01+0.j]],

       [[ 2.70598050e-01+0.j,  6.53281482e-01+0.j, -2.70598050e-01+0.j,
          6.53281482e-01+0.j],
        [-6.53281482e-01+0.j, -2.70598050e-01+0.j, -6.53281482e-01+0.j,
          2.70598050e-01+0.j],
        [ 6.53281482e-01+0.j, -2.70598050e-01+0.j, -6.53281482e-01+0.j,
         -2.70598050e-01+0.j],
        [ 2.70598050e-01+0.j, -6.53281482e-01+0.j,  2.70598050e-01+0.j,
          6.53281482e-01+0.j]]])
    
    is_equal = False 
    for j, qc in enumerate(qcs):
        qc = qc.copy()
        qc.remove_final_measurements()
        is_equal = False
        for op in Ops_2qb:
            if np.isclose( np.linalg.norm( Operator(qc).to_matrix() - op ), 0 ):
                is_equal = True
        if not is_equal:
            print( 'El circuito {} esta incorrecto'.format(j) )
            break

    if is_equal:
        print('Felicitaciones, tu solución es correcta!')

def test_3c( key, alice_random_trits, bob_random_trits  ):
    len_key = 0
    num_trits = len(alice_random_trits)
    for j in range(num_trits):
        a = alice_random_trits[j]
        b = bob_random_trits[j]
        if (a==1 and b==0) or (a==2 and b==1):
            len_key = len_key + 1 

    if len(key) == len_key:
        print('Felicidades, tu clave es segura')
    else:
        print('La longitud de tu clave es incorrecta')