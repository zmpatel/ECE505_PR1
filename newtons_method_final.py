import math
import sympy as sp

# Newton's method 
# Newton step: X_{k+1} = X_k - H^{-1}(X_k) * grad(f)(X_k)

x1, x2, x3 = sp.symbols('x1 x2 x3')

#Error & Iteration values
e = 10e-5
itmax = 1000

# **2D (2 variables)**

#f'(x)
def derivative(f,x,y):
    dx1 = sp.diff(f, x1).subs([(x1,x), (x2,y)])
    dx2 = sp.diff(f, x2).subs([(x1,x), (x2,y)])

    M = sp.Matrix([dx1, dx2])
    return M

# f''(x)
def second_derivative(f,x,y):
    dx1x1 = sp.diff(f, x1, x1).subs([(x1,x), (x2,y)])
    dx1x2 = sp.diff(f, x1, x2).subs([(x1,x), (x2,y)])
    dx2x1 = sp.diff(f, x2, x1).subs([(x1,x), (x2,y)])
    dx2x2 = sp.diff(f, x2, x2).subs([(x1,x), (x2,y)])
    
    M = sp.Matrix([[dx1x1, dx1x2], [dx2x1, dx2x2]])
    return M

# Solve H * s = g for s, then update x = x - s
# x_{k+1} = x_k - f'(x)/f''(x)
def newton_step_2(f,x,y):
    G = derivative(f,x,y)
    H = second_derivative(f,x,y)

    det = H.det()
    
    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")
    
    s1, s2 = H.LUsolve(G)
    print(f"search direction: [{s1:.4f}, {s2:.4f}]")
    print(f"step length: ||s|| = {math.sqrt(s1**2 + s2**2):.4f}\n")
    return x - s1, y - s2 


#calculate ||f''(x)|| / (1+|f'(x)|)
def comp2(f,x,y):
    G = derivative(f,x,y)
    H = second_derivative(f,x,y)

    det = H.det()

    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")
    
    s1, s2 = H.LUsolve(G)
    return math.sqrt(s1**2 + s2**2) / (1 + math.sqrt(G[0]**2 + G[1]**2))

#**3D (3 variables)**

#f'(x)
def derivative_3(f,x,y,z):
    dx1 = sp.diff(f, x1).subs([(x1,x), (x2,y), (x3,z)])
    dx2 = sp.diff(f, x2).subs([(x1,x), (x2,y), (x3,z)])
    dx3 = sp.diff(f, x3).subs([(x1,x), (x2,y), (x3,z)])
    #return gradient vector
    M = sp.Matrix([dx1, dx2, dx3])
    return M

#f''(x)
def second_derivative_3(f,x,y,z):
    dx1x1 = sp.diff(f,x1,x1).subs([(x1,x), (x2,y), (x3,z)])
    dx1x2 = sp.diff(f,x1,x2).subs([(x1,x), (x2,y), (x3,z)])
    dx1x3 = sp.diff(f,x1,x3).subs([(x1,x), (x2,y), (x3,z)])
    dx2x1 = sp.diff(f,x2,x1).subs([(x1,x), (x2,y), (x3,z)])
    dx2x2 = sp.diff(f,x2,x2).subs([(x1,x), (x2,y), (x3,z)])
    dx2x3 = sp.diff(f,x2,x3).subs([(x1,x), (x2,y), (x3,z)])
    dx3x1 = sp.diff(f,x3,x1).subs([(x1,x), (x2,y), (x3,z)])
    dx3x2 = sp.diff(f,x3,x2).subs([(x1,x), (x2,y), (x3,z)])
    dx3x3 = sp.diff(f,x3,x3).subs([(x1,x), (x2,y), (x3,z)])
    # Return the Hessian matrix 
    
    M = sp.Matrix([[dx1x1, dx1x2, dx1x3], [dx2x1, dx2x2, dx2x3], [dx3x1, dx3x2, dx3x3]])
    return M

#newtons step 3
def newton_step_3(f,x,y,z):
    G = derivative_3(f,x,y,z)
    H = second_derivative_3(f,x,y,z)

    det = H.det()
    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")

    s1, s2, s3 = H.LUsolve(G)
    print(f"search direction: [{s1:.4f}, {s2:.4f},{s3:.4f}]")
    print(f"step length: ||s|| = {math.sqrt(s1**2 + s2**2 + s3**2):.4f}\n")
    return x - s1, y - s2, z - s3


def comp3(f,x,y,z):
    G = derivative_3(f,x,y,z)
    H = second_derivative_3(f,x,y,z)
    
    det = H.det()
    
    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")

    s1, s2, s3 = H.LUsolve(G)
    return math.sqrt(s1**2 + s2**2 + s3**2) / (1 + math.sqrt(G[0]**2 + G[1]**2 + G[2]**2))


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
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    #print(f"problem {prob}:")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}\n")

    try:
        while (it < itmax and comp2(f,x,y) > e):
            it +=1
            x,y = newton_step_2(f,x,y)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}")

#Looping through 3 var functions (i.e. the one lol)
for function in functions_3:
    f, (x, y, z) = function, functions_3[function]
    it = 0
    prob += 1
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")
    try:
        while (it < itmax and comp3(f,x,y,z) > e):
            it +=1
            x,y,z = newton_step_3(f,x,y,z)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")







    


