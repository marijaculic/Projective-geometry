import numpy as np
import math
from numpy import linalg

#############################################
#1)
def Euler2A(phi, theta, psi):

    R_z = np.array([[math.cos(psi), -math.sin(psi), 0],
                    [math.sin(psi), math.cos(psi), 0],
                    [0, 0, 1]])

    R_y = np.array([[math.cos(theta), 0, math.sin(theta)],
                    [0, 1, 0],
                    [-math.sin(theta), 0, math.cos(theta)]])

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(phi), -math.sin(phi)],
                    [0, math.sin(phi), math.cos(phi)]])

    A = (R_z.dot(R_y)).dot(R_x)
    return A  #A je matrica kompozicije sopstvenih rotacija


#A = Euler2A(-1*math.atan2(8, 3), -1*math.asin(5/7) , math.atan2(3, 1))
A = Euler2A(0,0,math.pi)
print("Euler2A: \n")
print(A)
print("\n")

#######################################################
#2)
def crossProduct(r,q):
    p = np.zeros((1,3))
    p[0][0] = r[1]*q[2] - r[2]*q[1]
    p[0][1] = -1 * (r[0]*q[2] - r[2]*q[0])
    p[0][2] =  r[0]*q[1] - r[1]*q[0]
    
    return p

def dotProduct(u,u_prim):
    return u[0]*u_prim[0] + u[1]*u_prim[1] + u[2]*u_prim[2]
    
def AxisAngle(A):
    uslov1 = A.dot(A.T).round(5) == np.eye(3)
    uslov1 = uslov1.all() #all() vraca True ako su svi elementi matrice jednaki True
    uslov2 = np.linalg.det(A) == 1.0
    
    if not uslov1 or not uslov2:
        print("Matrica A nije matrica kretanja")
        return -1,-1
    
    #X = A-lambda*E
    #p je sopstveni vektor za lambda = 1
    
    X = A - np.eye(3)
    
    r = np.array([X[0][0],X[0][1],X[0][2]])
    q = np.array([X[1][0],X[1][1], X[1][2]])
    
    p = crossProduct(r,q)
    #ako su r i q linearno zavisni, uzimamo kao drugu jnu onu koju smo odbacili
    if p[0][0] == 0 and p[0][1] == 0 and p[0][2] == 0:
        q = np.array(X[2][0],X[2][1], X[2][2])
        
    p = crossProduct(r,q)
        
    #jedinicni vektor p:
    norm_p = math.sqrt(p[0][0]**2 + p[0][1]**2 + p[0][2]**2)
    p = p * (1/norm_p)
    
    #proizvoljan vektor u, normalan na vektor p
    #uzmemo npr. za u, vektor r i normiramo
    norm_r = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    u = r * (1/norm_r)
    
    u_prim = A.dot(u)
    
    cos_phi = dotProduct(u,u_prim)
    phi = math.acos(cos_phi)
    
    return p, phi
        

p,phi1 = AxisAngle(A)
print("AxisAngle: \n")
print("Prava: ", p)
print("Ugao: ", phi1)
print("\n")

#######################################################
#3)

def Rodrigez(p,phi):
    # Provera da li je vektor p jedinicni
    uslov = math.ceil(p[0][0]**2 + p[0][1]**2 + p[0][2]**2) == 1.0
    
    if not uslov:
        print("Vektor p nije jedinicni")
        #print(p[0][0]**2 + p[0][1]**2 + p[0][2]**2)
        return -1

    X = p.T.dot(p)  #ppT

    Y = np.eye(3) - X #E-ppT

    R = X + math.cos(phi)*Y

    p_x = np.array([[0, -p[0][2], p[0][1]],
                    [p[0][2], 0, -p[0][0]],
                    [-p[0][1], p[0][0], 0]])

    R = R + math.sin(phi)*p_x

    return R

R = Rodrigez(p,phi1)
print("Rodrigez: \n")
print(R)
print("\n")
#dobija se matrica iz prvog dela 1)

############################################################
#4)

def A2Euler(A):

    # Provera da li je matrica A ortogonalna
    uslov = np.linalg.det(A) == 1.0
    if not uslov:
        print("Matrica A nije ortogonalna!")
        return -1, -1, -1

    a = A[2][0]

    if a == 1:
        theta = -math.pi / 2
        phi = 0
        psi = math.atan2(-A[0][1], A[1][1])
    elif a == -1:
        theta = math.pi / 2
        phi = 0
        psi = math.atan2(-A[0][1], A[1][1])
    else:
        theta = math.asin(-a)
        phi = math.atan2(A[2][1], A[2][2])
        psi = math.atan2(A[1][0], A[0][0])
    
    return phi, theta, psi

phi, theta, psi = A2Euler(A)
print("A2Euler: \n")
print("Phi: ", phi) 
print("Theta: ", theta)
print("Psi: ", psi)
print("\n")
# dobijaju se pocetni uglovi (osim u slucaju gimbal lock-a)
    
#provera:
#print(-1*math.atan2(1,4))
#print(-1*math.asin(8/9))
#print(math.atan2(4,1))

###########################################################
#5)

def AxisAngle2Q(p, phi):
    # Provera da li je vektor p jedinicni
    uslov = math.ceil(p[0][0]**2 + p[0][1]**2 + p[0][2]**2) == 1.0
    if not uslov:
        print("Vektor p nije jedinicni!")
        norm = math.sqrt(p[0][0]**2 + p[0][1]**2 + p[0][2]**2)
        p = p * (1/norm)

    w = math.cos(phi/2)

    r = math.sin(phi/2) * p  #r je [x,y,z] 

    q = np.append(r,w)

    return q  

q = AxisAngle2Q(np.array([np.array([1,0,0])]), 0)
print("AxisAngle2Q: \n")
print(q)
print("\n")

##################################################################
#6)

def Q2AxisAngle(q):
    #Provera da li je kvaternion jedinicni
    uslov = (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2).round(3) == 1.0
    if not uslov:
        print("Kvaternion nije jedincni!")
        norm = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
        q = q * (1/norm)

    w = q[3]
    if w < 0:        #zelimo da phi pripada [0,pi]
        q = -q       

    phi = 2*math.acos(w)

    if abs(w) == 1:
        p = np.array([1, 0, 0])
    else:
        pomocni = np.array([q[0], q[1], q[2]])
        norm_pomocni = math.sqrt(pomocni[0]**2 + pomocni[1]**2 + pomocni[2]**2)
        p = pomocni * (1/norm_pomocni)

    return p, phi
    
p, phi = Q2AxisAngle(q)
print("Q2AxisAngle: \n")
print(p, phi) # rezultat je isti kao i u 2) 
    
