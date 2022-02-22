import numpy as np
from numpy import linalg as la

np.set_printoptions(suppress = True)

# 1)
def ParametriKamere(T):
    
    # pozicija kamere: C(c1, c2, c3, c4) trazimo iz jednakosti TC = 0
    
    #determinante minora:
    c1 = la.det([[T[0][1], T[0][2], T[0][3]],
                 [T[1][1], T[1][2], T[1][3]],
                 [T[2][1], T[2][2], T[2][3]]])
    
    c2 = -la.det([[T[0][0], T[0][2], T[0][3]],
                  [T[1][0], T[1][2], T[1][3]],
                  [T[2][0], T[2][2], T[2][3]]])
    
    c3 = la.det([[T[0][0], T[0][1], T[0][3]],
                 [T[1][0], T[1][1], T[1][3]],
                 [T[2][0], T[2][1], T[2][3]]])
    
    c4 = -la.det([[T[0][0], T[0][1], T[0][2]],
                  [T[1][0], T[1][1], T[1][2]],
                  [T[2][0], T[2][1], T[2][2]]])
    
    #normiramo koordinate
    c1 = c1/c4;
    c2 = c2/c4;
    c3 = c3/c4;
    c4 = 1.0
    
    C = np.array([c1,c2,c3])
    
    T0 = np.array(T[:, :3])  #samo bez poslednje kolone
    
    if la.det(T0) < 0:
        T0 = np.array(-T[:, :3])
    
    # nadjemo T0^-1
    # QR dekompozicija
    # K = R^-1 ili K = T0Q jer KA=T0=R^-1Q^-1, pomnozimo sa Q => T0Q = R^-1
    # A = Q^-1 ili A = Q^t jer je QQ^t = E
    
    Q,R = la.qr(la.inv(T0))
    
    #zelimo da svi elementi na dijagonali matrice R budu pozitivni
    #ukoliko neki nije, mnozimo sa -1 odgovarajucu vrstu matrice R i kolonu matrice Q
    
    if R[0][0] < 0:
        R[0] = -R[0]
        Q[:,0] = -Q[:,0]
    if R[1][1] < 0:
        R[1] = -R[1]
        Q[:,1] = -Q[:,1]
    if R[2][2] < 0:
        R[2] = -R[2]
        Q[:,2] = -Q[:,2]
        
    A = Q.T
    K = T0.dot(Q)
    
    #da bude k33 = 1
    K = K / K[2][2]
    
    return K, A, C
    
n = 9 #indeks: 269/2018
T = np.array([[5, -1-2*n, 3, 18-3*n],
             [0, -1, 5, 21],
             [0, -1, 0, 1]])

K, A, C = ParametriKamere(T)
print("K = \n ", K)  #matrica kalibracije
print("A = \n ", A)  #matrica orijentacije kamere
print("C = \n ", C)  #pozicija kamere 


############################################################################
# 2)

#tacke prostora:
M1 = np.array([460, 280, 250, 1])
M2 = np.array([50, 380, 350, 1])
M3 = np.array([470, 500, 100, 1])
M4 = np.array([380, 630, 50*n, 1])
M5 = np.array([30*n, 290, 0, 1])
M6 = np.array([580, 0, 130, 1])

#njihove projekcije:
M1p = np.array([288, 251, 1])
M2p = np.array([79, 510, 1])
M3p = np.array([470, 440, 1])
M4p = np.array([520, 590, 1])
M5p = np.array([365, 388, 1])
M6p = np.array([365, 20, 1])

originali = np.array([M1, M2, M3, M4, M5, M6])
projekcije = np.array([M1p, M2p, M3p, M4p, M5p, M6p])

# na osnovu svakog para originala i odgovarajuce slike (ima ih najmanje 6) odredjuemo dve jednacine date matricom 2x12
# pa ukupna matrica ima format 12x12

def CameraDLP(originali, projekcije):
    
    #uzimamo 4 koordinate prve originalne tacke M1
    x = originali[0][0]
    y = originali[0][1]
    z = originali[0][2]
    t = originali[0][3]

    #uzimamo 3 koordinate prve tacke M1 iz projekcije
    u = projekcije[0][0]
    v = projekcije[0][1]
    w = projekcije[0][2]

    A = np.array([
        [0, 0, 0, 0, -w*x, -w*y, -w*z, -w*t, v*x, v*y, v*z, v*t],
        [w*x, w*y, w*z, w*t, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u*t]
    ])

    for i in range(1, len(originali)):
        x = originali[i][0]
        y = originali[i][1]
        z = originali[i][2]
        t = originali[i][3]

        u = projekcije[i][0]
        v = projekcije[i][1]
        w = projekcije[i][2]

        prva_vrsta = np.array([0, 0, 0, 0, -w*x, -w*y, -w*z, -w*t, v*x, v*y, v*z, v*t])
        druga_vrsta = np.array([w*x, w*y, w*z, w*t, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u*t])

        A = np.vstack((A, prva_vrsta))
        A = np.vstack((A, druga_vrsta))

    # SVD dekompozicija
    U,D,V = np.linalg.svd(A) 
    T = V[-1].reshape(3, 4)   #poslednja vrsta matrice V^T
    
    return T
    
    
T = CameraDLP(originali, projekcije)
T = T / T[0][0] #da bude t11 = 1
print("T = \n", T)  #matrica projektovanja
