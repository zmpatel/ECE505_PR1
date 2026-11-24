#BFGS Quasi-Newton Method
#B0 = I
#xk+1 = xk+akpk
#pk = -B0^-1*f'(x)
#ak = 1, ak = 1/2 ak if f(xk + akpk) > f(xk)
#Bk+1 = Bk - ((Bk*sk)*(Bk*sk)^T)/(sk^T*Bk*sk)) + ((yk*yk^T)/(yk^T*sk)))
#Sk = xk+1 - xk
#Yk = f'(xk+1) - f'(xk)

import math
import sympy as sp

x1, x2, x3 = sp.symbols('x1 x2 x3')

#Error & Iteration values for TB #MOVE DOWN LATER
e = 10e-8
itmax = 1000

# **2D (2 variables)**

#f'(x)
def derivative(f,x,y):
    dx1 = sp.diff(f, x1).subs([(x1,x), (x2,y)])
    dx2 = sp.diff(f, x2).subs([(x1,x), (x2,y)])

    M = sp.Matrix([dx1, dx2])
    return M


def quasi_newton_bfgs_2(f,x,y,Bk):
    G = derivative(f,x,y)
    p = -Bk.inv() * G
    p0 = p[0]
    p1 = p[1]
    

    ak = 1.0

    f_current = f.subs([(x1,x), (x2,y)])
    while ak > 1e-12 and f.subs([(x1,x + ak*p0), (x2,y + ak*p1)]) > f_current:
            ak /= 2
    
    x_new = x+ak*p0
    y_new = y+ak*p1

    #Bk+1 = Bk - ((Bk*sk)*(Bk*sk)^T)/(sk^T*Bk*sk)) + ((yk*yk^T)/(yk^T*sk)))
    #Sk = xk+1 - xk
    #Yk = f'(xk+1) - f'(xk)

    G_new = derivative(f,x_new,y_new)

    sk = sp.Matrix([x_new-x, y_new-y])
    yk = G_new - G

    #update Bk

    Bk = Bk - ((Bk*sk)*(Bk*sk).T)/(sk.T*Bk*sk)[0] + ((yk*yk.T)/(yk.T*sk)[0])
    

    return x_new, y_new, Bk



#calculate ||f'(x)|| / (1+|f(x)|)
def comp2(f,x,y):
    G = derivative(f,x,y)
    
    return math.sqrt(G[0]**2 + G[1]**2) / (1 + math.sqrt(f.subs([(x1,x), (x2,y)])**2))

#**3D (3 variables)**

#f'(x)
def derivative_3(f,x,y,z):
    dx1 = sp.diff(f, x1).subs([(x1,x), (x2,y), (x3,z)])
    dx2 = sp.diff(f, x2).subs([(x1,x), (x2,y), (x3,z)])
    dx3 = sp.diff(f, x3).subs([(x1,x), (x2,y), (x3,z)])
    #return gradient vector
    M = sp.Matrix([dx1, dx2, dx3])
    return M

#Quasi-Newton BFGS 3D

def quasi_newton_bfgs_3(f,x,y,z,Bk):
    G = derivative_3(f,x,y,z)
    p = -Bk.inv() * G

    p0 = p[0]
    p1 = p[1]
    p2 = p[2]

    ak = 1.0

    f_current = f.subs([(x1,x), (x2,y), (x3,z)])
    while ak > 1e-12 and f.subs([(x1,x + ak*p0), (x2,y + ak*p1), (x3,z+ak*p2)]) > f_current:
        ak /= 2
    
    x_new = x+ak*p0
    y_new = y+ak*p1
    z_new = z+ak*p2

    #Bk+1 = Bk - ((Bk*sk)*(Bk*sk)^T)/(sk^T*Bk*sk)) + ((yk*yk^T)/(yk^T*sk)))
    #Sk = xk+1 - xk
    #Yk = f'(xk+1) - f'(xk)

    G_new = derivative_3(f,x_new,y_new,z_new)

    sk = sp.Matrix([x_new-x, y_new-y, z_new-z])
    yk = G_new - G

    #update Bk
    Bk = Bk - ((Bk*sk)*(Bk*sk).T)/(sk.T*Bk*sk)[0] + ((yk*yk.T)/(yk.T*sk)[0])

    return x_new, y_new, z_new, Bk


def comp3(f,x,y,z):
    G = derivative_3(f,x,y,z)

    return math.sqrt(G[0]**2 + G[1]**2 + G[2]**2) / (1 + math.sqrt(f.subs([(x1,x), (x2,y), (x3,z)])**2))


#***********TEST BENCH************

#TEST FUNCTIONS FROM 10d.
#functions dictionary with function as key and initial guess as value

#functions w/ 2 variables
#for final test c = 1, 10, 100
functions_2 = { 5*x1**4 + 6*x2**4 - 6*x1**2 + 2*x1*x2 + 5*x2**2 + 15*x1 - 7*x2 + 13 : (1.0,1.0),
                x1**2 + 2*x2**2 - 2*x1*x2 - 2*x2 : (0.0,0.0),
                100*(x2 - x1**2)**2 + (1 - x1)**2 : (-1.2,1.0),
                (x1+x2)**4 + x2**2 : (2.0,-2.0),
                (x1-1)**2 + (x2-1)**2 + 100*(x1**2 + x2**2 - 0.25)**2 : (1.0,-1.0)
}

#functions w/ 3 variables
functions_3 = {
                x1**2 + x2**2 + x3**2 : (1.0,1.0,1.0)
}

prob = 0; #problem #
#Looping through 2 var functions
for function in functions_2:
    f, (x, y) = function, functions_2[function]
    it = 0
    prob += 1
    Bk = sp.eye(2) #B0 = I
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    #print(f"problem {prob}:")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}\n")

    try:
        while (it < itmax and comp2(f,x,y) > e):
            it +=1

            x,y,Bk = quasi_newton_bfgs_2(f,x,y,Bk)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}, Bk = {Bk.evalf(4)}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}")

#Looping through 3 var functions (i.e. the one lol)
for function in functions_3:
    f, (x, y, z) = function, functions_3[function]
    it = 0
    prob += 1
    Bk = sp.eye(3) #B0 = I
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")
    try:
        while (it < itmax and comp3(f,x,y,z) > e):
            it +=1
            x,y,z,Bk = quasi_newton_bfgs_3(f,x,y,z,Bk)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}, Bk = {Bk.evalf(4)}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")







    





