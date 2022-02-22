#include <iostream>
#include <cmath>

using namespace std;

struct Tacka {
    double x,y;
    double z = 1;
};

void vektorski_proizvod(Tacka A, Tacka B, Tacka& r){
    r.x = A.y*B.z - A.z*B.y;
    r.y = -A.x*B.z + A.z*B.x;
    r.z = A.x*B.y - A.y*B.x;
}

void nevidljivo(Tacka A, Tacka B, Tacka D, Tacka A1, Tacka B1, Tacka C1, Tacka D1){
    
    //racunamo Xb
    
    Tacka AA1;
    vektorski_proizvod(A,A1,AA1);
    
    Tacka BB1;
    vektorski_proizvod(B,B1,BB1);
    
    Tacka DD1;
    vektorski_proizvod(D,D1,DD1);
    
    Tacka Xb1;
    vektorski_proizvod(AA1,BB1,Xb1);
    
    Tacka Xb2;
    vektorski_proizvod(AA1,DD1,Xb2);
    
    Tacka Xb3;
    vektorski_proizvod(BB1,DD1,Xb3);
    
    
    Tacka Xb;
    Xb.x = double(Xb1.x + Xb2.x + Xb3.x)/3;
    Xb.y = double(Xb1.y + Xb2.y + Xb3.y)/3;
    Xb.z = double(Xb1.z + Xb2.z + Xb3.z)/3;
    
    //racunamo Yb
    
    Tacka AB;
    vektorski_proizvod(A,B,AB);
    
    Tacka A1B1;
    vektorski_proizvod(A1,B1,A1B1);
    
    Tacka D1C1;
    vektorski_proizvod(D1,C1,D1C1);
    
    
    Tacka Yb1;
    vektorski_proizvod(AB,A1B1,Yb1);
    
    Tacka Yb2;
    vektorski_proizvod(AB,D1C1,Yb2);
    
    Tacka Yb3;
    vektorski_proizvod(A1B1,D1C1,Yb3);
    
    Tacka Yb;
    Yb.x = double(Yb1.x + Yb2.x + Yb3.x)/3;
    Yb.y = double(Yb1.y + Yb2.y + Yb3.y)/3;
    Yb.z = double(Yb1.z + Yb2.z + Yb3.z)/3;
    
    //racunamo Zb
    
    Tacka DA;
    vektorski_proizvod(D,A,DA);
    
    Tacka C1B1;
    vektorski_proizvod(C1,B1,C1B1);
    
    Tacka D1A1;
    vektorski_proizvod(D1,A1,D1A1);
    
    
    Tacka Zb1;
    vektorski_proizvod(DA,C1B1,Zb1);
    
    Tacka Zb2;
    vektorski_proizvod(DA,D1A1,Zb2);
    
    Tacka Zb3;
    vektorski_proizvod(C1B1,D1A1,Zb3);
    
    Tacka Zb;
    Zb.x = double(Zb1.x + Zb2.x + Zb3.x)/3;
    Zb.y = double(Zb1.y + Zb2.y + Zb3.y)/3;
    Zb.z = double(Zb1.z + Zb2.z + Zb3.z)/3;
    
    
    //racunamo tacku C
    
    Tacka C1Xb;
    vektorski_proizvod(C1,Xb,C1Xb);
    
    Tacka BZb;
    vektorski_proizvod(B,Zb,BZb);
    
    //Tacka DYb;
    //vektorski_proizvod(D,Yb,DYb);
    
    Tacka Cp;
    vektorski_proizvod(C1Xb,BZb,Cp);
    
    //Tacka Cd;
    //vektorski_proizvod(C1Xb,DYb,Cd);
    
    //Tacka Ct;
    //vektorski_proizvod(BZb,DYb,Ct);
    /*
    Tacka C;
    C.x = double(Cp.x + Cd.x + Ct.x)/3;
    C.y = double(Cp.y + Cd.y + Ct.y)/3;
    C.z = double(Cp.z + Cd.z + Ct.z)/3;
    */
    cout << "(" << ceil(Cp.x/Cp.z) << ", " << ceil(Cp.y/Cp.z) << ")" << endl;
}

int main(){
    Tacka A,B,D,A1,B1,C1,D1; //trazi se nevidljiva tacka C
    
    A.x = 292;  A.y = 517;
    B.x = 595;  B.y = 301;
    D.x = 157;  D.y = 379;
    A1.x = 304; A1.y = 295;
    B1.x = 665; B1.y = 116;
    C1.x = 509; C1.y = 43;
    D1.x = 135; D1.y = 163;
    
    /*
    
    A.x = 449;   A.y = 1685;
    B.x = 1399;  B.y = 1175;
    D.x = 93;    D.y = 1137;
    A1.x = 459;  A1.y = 1475;
    B1.x = 1431; B1.y = 995;
    C1.x = 903;  C1.y = 767;
    D1.x = 75;   D1.y = 977;
    */
    nevidljivo(A,B,D,A1,B1,C1,D1);
    
    return 0;
}

