import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Leva slika
L1 = [814, 110, 1]
L2 = [951, 161, 1]
L3 = [988, 124, 1]
L4 = [854, 79, 1]
L5 = [790, 305, 1]
L6 = [910, 356, 1]
L7 = [948, 319, 1]
L8 = [0, 0, 1]
L9 = [321, 345, 1]
L10 = [453, 370, 1]
L11 = [508, 272, 1]
L12 = [385, 250, 1]
L13 = [366, 559, 1]
L14 = [476, 585, 1]
L15 = [527, 487, 1]
L16 = [0, 0, 1]
L17 = [136, 554, 1]
L18 = [433, 761, 1]
L19 = [813, 379, 1]
L20 = [545, 251, 1]
L21 = [175, 657, 1]
L22 = [449, 859, 1]
L23 = [806, 488, 1]
L24 = [0, 0, 1]

# Desna slika
R1 = [911, 446, 1]
R2 = [808, 560, 1]
R3 = [915, 613, 1]
R4 = [1012, 491, 1]
R5 = [0, 0, 1]
R6 = [774, 770, 1]
R7 = [861, 815, 1]
R8 = [955, 704, 1]
R9 = [297, 74, 1]
R10 = [250, 122, 1]
R11 = [369, 136, 1]
R12 = [413, 88, 1]
R13 = [0, 0, 1]
R14 = [288, 324, 1]
R15 = [394, 343, 1]
R16 = [434, 287, 1]
R17 = [0, 0, 1]
R18 = [137, 320, 1]
R19 = [524, 526, 1]
R20 = [738, 347, 1]
R21 = [0, 0, 1]
R22 = [161, 430, 1]
R23 = [528, 644, 1]
R24 = [736, 458, 1]

################################################ Rekonstrukcija skrivenih tacaka ################################################

#za skrivene tacke leve slike
L8 = np.cross(np.cross(np.cross(np.cross(L5, L6), np.cross(L1, L2)), L7), 
              np.cross(np.cross(np.cross(L2, L6), np.cross(L3, L7)), L4))
L8 = L8 / L8[2]
L8 = [np.round(L8[0]), np.round(L8[1]), L8[2]]
#print("L8:  ", L8)

L16 = np.cross(np.cross(np.cross(np.cross(L13, L14), np.cross(L9, L10)), L15), 
               np.cross(np.cross(np.cross(L9, L13), np.cross(L10, L14)), L12))
L16 = L16 / L16[2]
L16 = [np.round(L16[0]), np.round(L16[1]), L16[2]]
#print("L16: ", L16)

L24 = np.cross(np.cross(np.cross(np.cross(L21, L22), np.cross(L19, L20)), L23), 
               np.cross(np.cross(np.cross(L17, L21), np.cross(L18, L22)), L20))
L24 = L24 / L24[2]
L24 = [np.round(L24[0]), np.round(L24[1]), L24[2]]
#print("L24: ", L24)

#za skrivene tacke desne slike
R5 = np.cross(np.cross(np.cross(np.cross(R1, R2), np.cross(R3, R4)), R6), 
              np.cross(np.cross(np.cross(R2, R6), np.cross(R3, R7)), R1))
R5 = R5 / R5[2]
R5 = [np.round(R5[0]), np.round(R5[1]), R5[2]]
#print("R5:  ", R5)

R13 = np.cross(np.cross(np.cross(np.cross(R10, R14), np.cross(R11, R15)), R9), 
              np.cross(np.cross(np.cross(R9, R10), np.cross(R11, R12)), R14))
R13 = R13 / R13[2]
R13 = [np.round(R13[0]), np.round(R13[1]), R13[2]]
#print("R13: ", R13)

R17 = np.cross(np.cross(np.cross(np.cross(R19, R20), np.cross(R23, R24)), R18), 
              np.cross(np.cross(np.cross(R18, R19), np.cross(R22, R23)), R20))
R17 = R17 / R17[2]
R17 = [np.round(R17[0]), np.round(R17[1]), R17[2]]
#print("R17: ", R17)

R21 = np.cross(np.cross(np.cross(np.cross(R19, R20), np.cross(R23, R24)), R22), 
              np.cross(np.cross(np.cross(R18, R19), np.cross(R22, R23)), R24))
R21 = R21 / R21[2]
R21 = [np.round(R21[0]), np.round(R21[1]), R21[2]]
#print("R21: ", R21)


left_image = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15, L16, L17, L18, L19, L20, L21, L22, L23, L24]
right_image = [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24]

# 8 tacaka na osnovu kojih odredjujemo fundamentalnu matricu F
L = np.array([L1, L2, L3, L4, L7, L8, L9, L11])
R = np.array([R1, R2, R3, R4, R7, R8, R9, R11])

#fja koja pravi jednacinu oblika y^T * F * x = 0 tj. r^T * F * l = 0
def equation(l, r):
    a1 = l[0]
    a2 = l[1]
    a3 = l[2]

    b1 = r[0]
    b2 = r[1]
    b3 = r[2]

    return np.array([a1*b1, a2*b1, a3*b1, a1*b2, a2*b2, a3*b2, a1*b3, a2*b3, a3*b3])

# pravimo matricu formata 8x9 (8 jednacina dobijenih iz korespodencija)
equations8 = []
for i in range(8):  
    equations8.append(equation(L[i], R[i]))

# SVD dekompozicija prethodne matrice (equations8)
U, D, V = LA.svd(equations8)

#koeficijenti matrice F su poslednja kolona matrice V
V = V[:][-1]
F = V.reshape(3,3)
print("Fundamentalna matrica:\n", F)

print("\nDeterminanta matrice F je ", LA.det(F)) # treba da bude blizu 0


# funkcija koja proverava da li je y^T * F * x = 0 za sve izabrane tacke

def proveri(x, y):
    tmp = np.dot(np.dot(y, F), x)
    return tmp

# provera uslova za svih 8 odgovarajucih tacaka
list = []
for i in range(0,8):
    element = proveri(L[i], R[i])
    list.append(element)
print("\nUslov vazi ako su svi elementi bliski nuli: \n", list)



################################################ trazimo epipolove ################################################

#za epipol e1 treba da resimo sistem F*e1 = 0 (to radimo SVD dekompozicijom)
U, DD, V = LA.svd(F)

e1 = V[:][-1] # treca vrsta matrice V 
print("\nEpipol e1: ", e1)
e1 = (1/e1[2]) * e1 # afine koordinate
print("\nAfine koordinate epipola e1: ", e1)

#za epipol e2 treba da resimo sistem F^T * e2 = 0
#a posto je F^T = (UDV^T)^T = VDU^T tj. ako je UDV dekompozicija od F, onda ce VDU biti dekompozicija od F^T
e2 = U.T[:][-1] # treca kolona matrice U (tj. treca vrsta od U.T)
print("\nEpipol e2: ", e2)
e2 = (1/e2[2]) * e2 # afine koordinate
print("\nAfine koordinate epipola e2: ", e2)



################################################ Triangulacija ################################################

# pravimo kanonsku matricu kamere T1 = [E|0]
T1 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]])
print("\nT1:\n", T1)

# pravimo matricu vektorskog mnozenja (kao kod Rodrigez formule)
E2 = np.array([[0, -e2[2], e2[1]],
               [e2[2], 0, -e2[0]],
               [-e2[1], e2[0], 0]])
#print("E2:\n", E2)

# pravimo matricu druge kamere T2 = [M|e2] , M = E2*F
tmp = E2.dot(F)
T2 = np.array([[tmp[0][0], tmp[0][1], tmp[0][2], e2[0]],
               [tmp[1][0], tmp[1][1], tmp[1][2], e2[1]],
               [tmp[2][0], tmp[2][1], tmp[2][2], e2[2]]])
print("\nT2:\n", T2)

# Za svaku tacku dobijamo sistem od 4 jednacine sa 4 homogene nepoznate
def equations(l, d):
    return np.array([l[1]*T1[2] - l[2]*T1[1],
                    -l[0]*T1[2] + l[2]*T1[0],
                    d[1]*T2[2] - d[2]*T2[1],
                    -d[0]*T2[2] + d[2]*T2[0]])

# Vracamo 3D koordinate tacke
def TriD_koo(l, r):
    U, D, V = LA.svd(equations(l, r))
    P = V[-1]     #poslednja vrsta
    P = P / P[3]  #pretvaramo u afine koordinate
    return P[:-1]

rekonstruisane = []
for i in range(len(left_image)):
    rekonstruisane.append(TriD_koo(left_image[i], right_image[i]))


# Zbog toga sto nismo radili normalizaciju (3. koordinata je mnogo manja od prve 2), mnozimo z koordinatu sa nekoliko stotina (npr. 400) 
tmp = np.eye(3)
tmp[2][2] = 400
rekonstruisane_400 = np.zeros((24,3))
for i in range(len(rekonstruisane)):
    rekonstruisane_400[i] = tmp.dot(rekonstruisane[i])

print("\nRekonstruisane tacke: \n", rekonstruisane_400)
tacke = rekonstruisane_400

################################################  crtanje  ################################################ 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([200, 800])
ax.set_ylim([200, 800])
ax.set_zlim([200, 800])

S = tacke[:8]
stranice = [[S[0],S[1],S[2],S[3]],     #stranica kutije 1234
            [S[4],S[5],S[6],S[7]],     #stranica kutije 5678
            [S[0],S[1],S[5],S[4]],     #stranica kutije 1265
            [S[2],S[3],S[7],S[6]],     #stranica kutije 3487
            [S[1],S[2],S[6],S[5]],     #stranica kutije 2376
            [S[4],S[7],S[3],S[0]]]     #stranica kutije 5841
ax.add_collection3d(Poly3DCollection(stranice, facecolors='red', linewidths=1, edgecolors='r', alpha=.25))

S = tacke[8:16]
stranice = [[S[0],S[1],S[2],S[3]],
            [S[4],S[5],S[6],S[7]], 
            [S[0],S[1],S[5],S[4]], 
            [S[2],S[3],S[7],S[6]], 
            [S[1],S[2],S[6],S[5]],
            [S[4],S[7],S[3],S[0]]]
ax.add_collection3d(Poly3DCollection(stranice, facecolors='yellow', linewidths=1, edgecolors='y', alpha=.25))

S = tacke[16:24]
stranice = [[S[0],S[1],S[2],S[3]],
            [S[4],S[5],S[6],S[7]], 
            [S[0],S[1],S[5],S[4]], 
            [S[2],S[3],S[7],S[6]], 
            [S[1],S[2],S[6],S[5]],
            [S[4],S[7],S[3],S[0]]]
ax.add_collection3d(Poly3DCollection(stranice, facecolors='blue', linewidths=1, edgecolors='b', alpha=.25))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
    
plt.gca().invert_yaxis()
plt.show()
