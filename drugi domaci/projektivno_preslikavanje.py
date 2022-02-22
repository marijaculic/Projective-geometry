import numpy as np
import math
import cv2

def nadji_matricu(tacke):
    matrica_sistema = np.array([
        [tacke[0][0],tacke[1][0],tacke[2][0]],
        [tacke[0][1],tacke[1][1],tacke[2][1]],
        [tacke[0][2],tacke[1][2],tacke[2][2]]
    ])
    
    #D = alfa*A + beta*B + gama*C
    #tacke_transponovano = np.transpose(tacke)
    #D = tacke_transponovano[:,-1]
    D= np.array([tacke[3][0],tacke[3][1],tacke[3][2]])
    
    rezultat = np.linalg.solve(matrica_sistema,D)
    
    alfa = rezultat[0]
    beta = rezultat[1]
    gama = rezultat[2]
    
    prva_kolona = np.array([alfa*tacke[0][0], alfa*tacke[0][1], alfa* tacke[0][2]])
    druga_kolona = np.array([beta*tacke[1][0],beta*tacke[1][1], beta*tacke[1][2]])
    treca_kolona = np.array([gama*tacke[2][0], gama*tacke[2][1], gama*tacke[2][2]])
    
    #P = np.transpose([prva_kolona,druga_kolona,treca_kolona])
    P = np.column_stack([prva_kolona,druga_kolona,treca_kolona])
    
    return P
    
    
def naivni_algoritam(originalne_tacke, preslikane_tacke):
    #nadjemo matricu P1
    P1 = nadji_matricu(originalne_tacke)

    #nadjemo matricu P2
    P2 = nadji_matricu(preslikane_tacke)

    #trazena matrica P = P2 * P1^-1
    P = np.dot(P2,np.linalg.inv(P1))

    return P
    
#################################################################

def DLT(originalne_tacke, preslikane_tacke):
    x1 = originalne_tacke[0][0]
    x2 = originalne_tacke[0][1]
    x3 = originalne_tacke[0][2]
    x1p = preslikane_tacke[0][0]
    x2p = preslikane_tacke[0][1]
    x3p = preslikane_tacke[0][2]
    
    A = np.array([
        [0,0,0, -x3p*x1, -x3p*x2, -x3p*x3, x2p*x1, x2p*x2, x2p*x3],
        [x3p*x1, x3p*x2, x3p*x3, 0,0,0, -x1p*x1, -x1p*x2, -x1p*x3]
        ])
    
    for i in range(1, len(originalne_tacke)):
        x1 = originalne_tacke[i][0]
        x2 = originalne_tacke[i][1]
        x3 = originalne_tacke[i][2]
        x1p = preslikane_tacke[i][0]
        x2p = preslikane_tacke[i][1]
        x3p = preslikane_tacke[i][2]
        
        prva_vrsta = np.array([0,0,0, -x3p*x1, -x3p*x2, -x3p*x3, x2p*x1, x2p*x2, x2p*x3])
        druga_vrsta = np.array([x3p*x1, x3p*x2, x3p*x3, 0,0,0, -x1p*x1, -x1p*x2, -x1p*x3])
        
        A = np.vstack((A, prva_vrsta))
        A = np.vstack((A, druga_vrsta))
        
    U,D,V = np.linalg.svd(A)
    
    #print(A)
    P = V[-1].reshape(3,3)   #poslednja vrsta matrice V^T
    
    return P
    
#######################################################################

def normalizacija(originalne_tacke):

    #teziste sa afinim:
    cx = sum([el[0]/el[2] for el in originalne_tacke]) / len(originalne_tacke)
    cy = sum([el[1]/el[2] for el in originalne_tacke]) / len(originalne_tacke)
    
    rastojanje = 0.0
    
    #translacija u koordinatni pocetak
    for i in range(0,len(originalne_tacke)):
        #svaku tacku pomeramo za (-cx,-cy)
        a = float(originalne_tacke[i][0]/originalne_tacke[i][2]) - cx
        b = float(originalne_tacke[i][1]/originalne_tacke[i][2]) - cy
    
        rastojanje = rastojanje + math.sqrt(a**2 + b**2)
        
    rastojanje = rastojanje / float(len(originalne_tacke))
    
    #prosecno rastojanje treba da bude koren iz 2
    #S je faktor skaliranja:
    S = float(math.sqrt(2))/rastojanje
    
    #T je matrica normalizacije (izveden proizvod S i G)
    T = np.array([[S,0,-S*cx], [0,S,-S*cy], [0,0,1]])
    
    return T

def modifikovani_DLT(originalne_tacke,preslikane_tacke):
    T = normalizacija(originalne_tacke)
    Tp = normalizacija(preslikane_tacke)
    
    #primenimo matricu normalizacije na koordinate tacaka:
    M = T.dot(np.transpose(originalne_tacke))
    Mp = Tp.dot(np.transpose(preslikane_tacke))
    
    M = np.transpose(M) 
    Mp = np.transpose(Mp) 
    
    Pp = DLT(M,Mp)
    
    P = (np.linalg.inv(Tp)).dot(Pp).dot(T)
    
    return P

#4 originalne tacke
#originalne_tacke = np.array([[-3,-1,1],[3,-1,1],[1,1,1],[-1,1,1]])
originalne_tacke = np.array([[2,0,1],[-2,1,1],[-1,-4,1],[0,2,1]])

#4 preslikane tacke
#preslikane_tacke = np.array([[-2,-1,1],[2,-1,1],[2,1,1],[-2,1,1]])
preslikane_tacke = np.array([[-2,1,1],[2,-1,1],[1,-2,1],[3,-1,1]])

print("1)\n")
P_naivni = naivni_algoritam(originalne_tacke, preslikane_tacke)
print("Naivni algoritam:")
P_naivni = (P_naivni / P_naivni[0][0]).round(5)
print(P_naivni)
print("\n")

P_dlt = DLT(originalne_tacke, preslikane_tacke)
print("DLT algoritam:")
P_dlt = (P_dlt / P_dlt[0][0]).round(5)
print(P_dlt)
print("\n")

#poredjenje naivnog i DLT algoritma:
print("poredjenje naivnog i DLT:")
P_dlt = (P_dlt / P_dlt[0][0]) * P_naivni[0][0]
print(P_dlt.round(5) == P_naivni.round(5))
print("\n")

P_modifikovano = modifikovani_DLT(originalne_tacke, preslikane_tacke)
print("modifikovani DLT algoritam:")
P_modifikovano = (P_modifikovano*1/P_modifikovano[0][0]).round(5)
print(P_modifikovano)
print("\n")

#poredjenje DLT i modifikovanog DLT algoritma:
print("poredjenje DLT i modifikovanog DLT algoritma:")
P_modifikovano = (P_modifikovano / P_modifikovano[0][0]) * P_dlt[0][0]
print(P_modifikovano.round(5) == P_dlt.round(5))
print("\n")


#DLT za vise od 4 korespodencije
#originalne_tacke = [[-3, -1, 1], [3, -1, 1], [1, 1, 1], [-1, 1, 1], [1, 2, 3],[-8, -2, 1]]
originalne_tacke = [[2, 0 ,1], [-2, 1, 1],[-1, -4, 1], [0, 2, 1], [2, 2 ,1]]

preslikane_tacke = [[-2, 1, 1], [2, -1, 1], [1, -2, 1], [3, -1, 1],[-12, 1, 1]]
#preslikane_tacke = [[-2, -1, 1], [2, -1, 1], [2, 1, 1], [-2, 1, 1],[2, 1, 4],[-16,-5,4]]

print("2)\n")
P_dlt_vise = DLT(originalne_tacke, preslikane_tacke)
print("DLT za vise od 4 korespodencije:")
P_dlt_vise = (P_dlt_vise / P_dlt_vise[0][0]).round(5)
print(P_dlt_vise)
print("\n")

#poredjenje sa naivnim:
#print("Poredjenje DLT-a za vise od 4 korespodencije sa naivnim:")
#P_dlt_vise = (P_dlt_vise / P_dlt_vise[0][0]) * P_naivni[0][0]
#print(P_dlt_vise.round(5) == P_naivni.round(5))
#print("\n")

#DLT modifikovani za vise od 4 korespodencije
P_modifikovano_vise = modifikovani_DLT(originalne_tacke, preslikane_tacke)
print("DLT modifikovani za vise od 4 korespodencije:")
P_modifikovano_vise = (P_modifikovano_vise / P_modifikovano_vise[0][0]).round(5)
print(P_modifikovano_vise)
print("\n")


#poredimo DLT modifikovani za vise od 4 korespodencije i naivni
#print("Poredjenje modifikovanog DLT-a za vise od 4 korespodencije sa naivnim:")
#P_modifikovano_vise = (P_modifikovano_vise / P_modifikovano_vise[0][0]) * P_naivni[0][0]
#print(P_modifikovano_vise.round(5) == P_naivni.round(5))
#print("\n")

###### 
#poredimo DLT i modifikovani DLT oba za vise od 4 korespodencije
print("Poredjenje DLT za vise od 4 korespodencije i modifikovanog DLT za vise od 4 korespodencije: ")
P_modifikovano_vise = (P_modifikovano_vise / P_modifikovano_vise[0][0]) * P_dlt_vise[0][0]
print(P_modifikovano_vise.round(5) == P_dlt_vise.round(5))
print("\n")




originalne_tacke = [[-1,-4,1],[-4,1,1],[-8,-5,1],[-1,0,1],[1,-2,1]]
preslikane_tacke = [[3,-3,1],[5,1,1],[6,0,1],[5,2,1],[3,-13,1]]

print("3)\n")
P_modifikovano_novo = modifikovani_DLT(originalne_tacke, preslikane_tacke)
print("modifikovani DLT algoritam:")
P_modifikovano_novo = (P_modifikovano_novo*1/P_modifikovano_novo[0][0]).round(5)
print(P_modifikovano_novo)
print("\n")

####################################################################


#### PROMENA KOORDINATA KOD DLT #####

originalne_tacke = [[-3,-1,1],[3,-1,1],[1,1,1],[-1,1,1],[1,2,3],[-8,-2,1]]
preslikane_tacke = [[-2,-1,1],[2,-1,1],[2,1,1],[-2,1,1],[2,1,4],[-16,-5,4]]

C1 = np.array([[0,1,2],[-1,0,3],[0,0,1]])
C2 = np.array([[1,-1,5],[1,1,-2],[0,0,1]])

originalne_tacke_nove = []
preslikane_tacke_nove = []

for i in range(0,len(originalne_tacke)):
    originalne_tacke_nove.append(np.dot(C1,originalne_tacke[i]))
    preslikane_tacke_nove.append(np.dot(C2,preslikane_tacke[i]))
    
originalne_tacke_nove = np.array(originalne_tacke_nove)
preslikane_tacke_nove = np.array(preslikane_tacke_nove)
    
#DLT algoritam

P_dlt = DLT(originalne_tacke,preslikane_tacke)  #sa skice je to F
P_dlt_novo = DLT(originalne_tacke_nove, preslikane_tacke_nove) #sa skice je to R

P_tmp = np.dot(np.linalg.inv(C2), P_dlt_novo)
P_tmp = np.dot(P_tmp, C1)   # HRG^-1
P_tmp = (P_dlt[0] / P_tmp[0]) * P_tmp  #provera da li je F = HRG^-1

print("poredjenje DLT algoritma nakon promene koordinata: ")
print(P_dlt.round(5) == P_tmp.round(5))
print("\n")
# => matrica nije ista, tj. DLT nije invarijantan u odnosu na promenu koordinata, tj. nije geometrijski


#modifikovani DLT 

P_dlt_mod = modifikovani_DLT(originalne_tacke,preslikane_tacke)
P_dlt_mod_novo = modifikovani_DLT(originalne_tacke_nove,preslikane_tacke_nove)

P_tmp = np.dot(np.linalg.inv(C2),P_dlt_mod_novo)
P_tmp = np.dot(P_tmp,C1)

print("poredjenje modifikovanog DLT algoritma nakon promene koordinata:")
print(P_dlt_mod.round(5) == P_tmp.round(5))
print("\n")

# => matrica je ista, tj. modifikovani DLT jeste invarijantan u odnosu na promenu koordinata, tj. geometrijski je



array = np.zeros((4,3),int)
counter = 0
def click(event,x,y,flags,param):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        array[counter] = x,y,1
        counter = counter + 1
        #print(x,y)
        

image = cv2.imread('zgrada.jpeg')
image = cv2.resize(image,(470,630))


def aplikacija():
    while True:
        
        if counter == 4:
            tacke = np.float32([array[0],array[1],array[2],array[3]])
            preslikane = np.float32([[238,546,1],[256,354,1],[463,363,1],[462,537,1]])
            M = naivni_algoritam(tacke, preslikane)
            final = cv2.warpPerspective(image,M,(470,630))
            cv2.imshow('Output',final)
            
            
        cv2.namedWindow('Input')
        cv2.setMouseCallback('Input',click)

        cv2.imshow('Input',image)
        cv2.waitKey(1)


aplikacija()
