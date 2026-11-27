import math
import sympy as sp

#Steepest Descent Method
#Xk+1 = xk + akpk
#pk = -f'(x)
#ak = backtracking line search w/ Armijo condition


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

#steepest descent 2
def steepest_descent_2(f,x,y):
    G = derivative(f,x,y)
    p0 = -G[0]
    p1 = -G[1]

    ak = 1.0
    f_current = float(f.subs({x1: x, x2: y}))
    s = p0 * G[0] + p1 * G[1]

    while ak > 1e-12 and float(f.subs({x1: x + ak * p0, x2: y + ak * p1})) > f_current + c * ak * s:
        ak /= 2

    x = x+ak*p0
    y = y+ak*p1

    print(f"Descent Direction: p = ({p0:.4f}, {p1:.4f})")
    print(f"Step Size: ak = {ak:.4f}")

    return x, y


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
    M = sp.Matrix([dx1, dx2, dx3])
    return M

#steepest descent 3
def steepest_descent_3(f,x,y,z):
    G = derivative_3(f,x,y,z)
    p0 = -G[0]
    p1 = -G[1]
    p2 = -G[2]

    ak = 1.0  # Initial step size
    f_current = float(f.subs({x1: x, x2: y, x3: z}))
    s = p0 * G[0] + p1 * G[1] + p2 * G[2]
    
    # Backtracking line search w/ Armijo condition
    while ak > 1e-12 and float(f.subs({x1: x + ak * p0, x2: y + ak * p1, x3: z + ak * p2})) > f_current + c * ak * s:
        ak /= 2.0
        
    # Update coordinates using the valid step size
    x_new = x + ak * p0
    y_new = y + ak * p1
    z_new = z + ak * p2

    print(f"Descent Direction: p = ({p0:.4f}, {p1:.4f}, {p2:.4f})")
    print(f"Step Size: ak = {ak:.4f}")
    
    return x_new, y_new, z_new


def norm3(f,x,y,z):
    G = derivative_3(f,x,y,z)

    return math.sqrt(G[0]**2 + G[1]**2 + G[2]**2) / (1 + math.sqrt(f.subs([(x1,x), (x2,y), (x3,z)])**2))


#***********TEST BENCH************

#TEST FUNCTIONS FROM 10d.
#functions dictionary with function as key and initial guess as value

#functions w/ 2 variables
#Problems 2,3,4
functions_2 = { 
                x1**2 + 2*x2**2 - 2*x1*x2 - 2*x2 : (0.0,0.0),
                100*(x2 - x1**2)**2 + (1 - x1)**2 : (-1.2,1.0),
                (x1+x2)**4 + x2**2 : (2.0,-2.0),
}

#functions w/ 3 variables
#problem 1
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
    try:
        while (it < itmax and norm3(f,x,y,z) > e):
            it +=1
            x,y,z = steepest_descent_3(f,x,y,z)
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
    #print(f"problem {prob}:")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}\n")

    try:
        while (it < itmax and norm2(f,x,y) > e):
            it +=1
            x,y = steepest_descent_2(f,x,y)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}")


#variable function C = 1,10,100
#Problem #5
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
            x,y = steepest_descent_2(f,x,y)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}")






    


