import math
import sympy as sp
#verified 

# Newton's method 
# Newton step: X_{k+1} = X_k - H^{-1}(X_k) * grad(f)(X_k)
#line search -> ak = 1, if f(x_k + ak * s_k) > f(x_k), then reduce ak=1/2ak until f decreases
#Implement Armijo condition:
    # f(xk+akpk) <= f(xk)+ c akpk^Tf'(xk)
    # c = 1e-4 (10^-4)

x1, x2, x3 = sp.symbols('x1 x2 x3')


#Error & Iteration values
e = 1e-5
itmax = 1000
c = 1e-4

# **2D (2 variables)**

#f'(x)
def derivative(f,x,y):
    dx1 = sp.diff(f, x1).subs([(x1,x), (x2,y)])
    dx2 = sp.diff(f, x2).subs([(x1,x), (x2,y)])

    M = sp.Matrix([dx1, dx2])
    return M

# f''(x)
def Hessian_2(f,x,y):
    H = sp.hessian(f, (x1, x2)).subs([(x1, x), (x2, y)])
    return H

# pk = -H^-1 * f'(x)
# x_{k+1} = x_k + ak * pk
#armijo's condition : f(xk+akpk) <= f(xk)+ c akpk^Tf'(xk)
def newton_step_2(f,x,y):
    G = derivative(f,x,y) #f'(x)
    H = Hessian_2(f,x,y) #f''(x)
    ak = 1.0

    det = H.det()
    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")

    pk = -H.LUsolve(G)
    pk0, pk1 = pk

    f_current = f.subs([(x1,x), (x2,y)])
    #armijo's condition: f(xk+akpk) <= f(xk)+ c akpk^Tf'(xk)
    s = pk0 * G[0] + pk1 * G[1]  # pk^T * f'(x)

    while ak > 1e-12 and f.subs([(x1,x + ak*pk0), (x2,y + ak*pk1)]) > f_current + c * ak * s:
        ak /= 2

    x_new = x + ak*pk0
    y_new = y + ak*pk1

    print(f"Descent direction: [{pk0:.4f}, {pk1:.4f}]")
    print(f"step length: ak = {ak:.4f}\n")
    return x_new, y_new


#calculate ||f'(x)|| / (1+|f(x)|)
def norm2(f,x,y):
    G = derivative(f,x,y)
    
    return math.sqrt(G[0]**2 + G[1]**2) / (1 + math.sqrt(f.subs([(x1,x), (x2,y)])**2))

#**3D (3 variables)**

#f'(x)
def derivative_3(f,x,y,z):
    dx1 = sp.diff(f, x1).subs([(x1,x), (x2,y), (x3,z)])
    dx2 = sp.diff(f, x2).subs([(x1,x), (x2,y), (x3,z)])
    dx3 = sp.diff(f, x3).subs([(x1,x), (x2,y), (x3,z)])
    #return gradient vector
    M = sp.Matrix([dx1, dx2, dx3]).subs([(x1,x), (x2,y), (x3,z)])

    return M

#f''(x)
def Hessian_3(f,x,y,z):
    H = sp.hessian(f, (x1, x2, x3)).subs([(x1, x), (x2, y), (x3, z)])
    return H

#newtons step 3

def newton_step_3(f,x,y,z):
    G = derivative_3(f,x,y,z)
    H = Hessian_3(f,x,y,z)
    ak = 1.0

    det = H.det()
    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")

    pk = -H.LUsolve(G)
    pk0, pk1, pk2 = pk

    s = pk0 * G[0] + pk1 * G[1] + pk2 * G[2]  # pk^T * f'(x)
    f_current = f.subs([(x1,x), (x2,y), (x3,z)])

    while ak > 1e-12 and f.subs([(x1,x + ak*pk0), (x2,y + ak*pk1), (x3,z + ak*pk2)]) > f_current + c * ak * s:
        ak /= 2

    x_new = x + ak*pk0
    y_new = y + ak*pk1
    z_new = z + ak*pk2

    print(f"search direction: [{pk0:.4f}, {pk1:.4f}, {pk2:.4f}]")
    print(f"step length: ak = {ak:.4f}\n")
    return x_new, y_new, z_new

# |f'(x)|/(1+ |f(x)|)
def norm3(f,x,y,z):
    G = derivative_3(f,x,y,z)

    return (math.sqrt(G[0]**2 + G[1]**2 + G[2]**2)/(1+math.sqrt(f.subs([(x1,x), (x2,y), (x3,z)])**2)))


#***********TEST BENCH************

#TEST FUNCTIONS FROM 10d.
#functions dictionary with function as key and initial guess as value

#functions w/ 2 variables
#for final test c = 1, 10, 100
functions_2 = { 
                x1**2 + 2*x2**2 - 2*x1*x2 - 2*x2 : (0.0,0.0),
                100*(x2 - x1**2)**2 + (1 - x1)**2 : (-1.2,1.0),
                (x1+x2)**4 + x2**2 : (2.0,-2.0),
}

#functions w/ 3 variables
functions_3 = {
                x1**2 + x2**2 + x3**2 : (1.0,1.0,1.0)
}

prob = 0; #problem #

#Looping through 3 var functions (i.e. the one lol)
for function in functions_3:
    f, (x, y, z) = function, functions_3[function]
    it = 0
    prob += 1
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")
    print(f"function: {f}\n")

    try:
        while (it < itmax and norm3(f,x,y,z) > e):
            it +=1
            x,y,z = newton_step_3(f,x,y,z)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")


#Looping through 2 var functions
for function in functions_2:
    f, (x, y) = function, functions_2[function]
    it = 0
    prob += 1
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    print(f"function: {f}\n")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}\n")

    try:
        while (it < itmax and norm2(f,x,y) > e):
            it +=1
            x,y = newton_step_2(f,x,y)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}")


#variable function C = 1,10,100
prob += 1
for C in [1, 10, 100]:
    f, (x, y) = (x1-1)**2 + (x2-1)**2 + C*(x1**2 + x2**2 - 0.25)**2, (1.0,-1.0)
    it = 0
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    print(f"C = {C}\n")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}\n")
    print(f"function: {f}\n")

    try:
        while (it < itmax and norm2(f,x,y) > e):
            it +=1
            x,y = newton_step_2(f,x,y)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}")







    


