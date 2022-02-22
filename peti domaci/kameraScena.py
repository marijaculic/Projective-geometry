import numpy as np
from numpy import linalg as la

np.set_printoptions(suppress = True)

#izmerene koordinate scene
M1 = np.array([233, 20, 54, 1])
M2 = np.array([140, 132, 54, 1])
M3 = np.array([12, 102, 203, 1])
M4 = np.array([68, 98, 4, 1])
M5 = np.array([230, 258, 76, 1])
M6 = np.array([204, 352, 7, 1])

#pikseli iz paint-a
M1p = np.array([20, 353, 1])
M2p = np.array([190, 405, 1])
M3p = np.array([258, 158, 1])
M4p = np.array([222, 365, 1])
M5p = np.array([210, 443, 1])
M6p = np.array([364, 618, 1])

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
print("T = \n", T.round(4))  #matrica projektovanja
