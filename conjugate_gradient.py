import math
import sympy as sp

#CONJUGATE GRADIENT METHOD
#ak = (f'(xk)^T * pk) / (pk^T * A * pk)
#xk+1 = xk + ak * pk
#pk = rk + beta * pk-1
#rk = b - A * xk
#beta = (rk^T * rk) / (rk-1^T * rk-1)


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

#f''(x) -> Hessian
def hessian_2(f,x,y):
    H = sp.hessian(f, (x1, x2)).subs([(x1,x), (x2,y)])
    return H

#steepest descent 2
def conjugate_grad_2(f,x,y):
    G = derivative(f,x,y)
    A = hessian_2(f,x,y)
    rk = -G

    pk = rk

    #     # Step size: ak = (rk^T * rk) / (pk^T * A * pk)
    # # Note: Since rk = -f'(xk) and pk is conjugate, -(f'(xk)^T * pk) equals (rk^T * rk)
    # ak = (rk.T * rk)[0] / (pk.T * A * pk)[0]

    ak = (rk.T * rk)[0] / (pk.T * A * pk)[0]

    # # Update position: x_{k+1} = x_k + ak * pk
    # xk = xk + ak * pk
    x_new, y_new = x,y + ak * pk

    # # New residual: r_{k+1} = -f'(x_{k+1})
    rk_next = -derivative(f,x_new,y_new)

    # # Beta: (r_{k+1}^T * r_{k+1}) / (rk^T * rk)
    beta = (rk_next.T * rk_next)[0] / (rk.T * rk)[0]

    # # Update search direction: p_{k+1} = r_{k+1} + beta * pk
    pk = rk_next + beta * pk
    rk = rk_next

    return x_new, y_new

    
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

def hessian_3(f,x,y,z):
    H = sp.hessian(f, (x1, x2,x3)).subs([(x1,x), (x2,y), (x3,z)])
    return H

#conjugate gradient 3
def conjugate_grad_3(f,x,y,z):
    G = derivative_3(f,x,y,z)
    A = hessian_3(f,x,y,z)
    rk = -G

    pk = rk

    #     # Step size: ak = (rk^T * rk) / (pk^T * A * pk)
    # # Note: Since rk = -f'(xk) and pk is conjugate, -(f'(xk)^T * pk) equals (rk^T * rk)
    # ak = (rk.T * rk)[0] / (pk.T * A * pk)[0]

    ak = (rk.T * rk)[0] / (pk.T * A * pk)[0]

    # # Update position: x_{k+1} = x_k + ak * pk
    # xk = xk + ak * pk
    x_new, y_new, z_new = x,y,z + ak * pk

    # # New residual: r_{k+1} = -f'(x_{k+1})
    rk_next = -derivative_3(f,x_new,y_new,z_new)

    # # Beta: (r_{k+1}^T * r_{k+1}) / (rk^T * rk)
    beta = (rk_next.T * rk_next)[0] / (rk.T * rk)[0]

    # # Update search direction: p_{k+1} = r_{k+1} + beta * pk
    pk = rk_next + beta * pk
    rk = rk_next

    return x_new, y_new, z_new


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
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    #print(f"problem {prob}:")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}\n")

    try:
        while (it < itmax and comp2(f,x,y) > e):
            it +=1
            x,y = conjugate_grad_2(f,x,y)
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
            x,y,z = conjugate_grad_3(f,x,y,z)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")







    



