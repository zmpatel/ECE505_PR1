import math
import sympy as sp

#CONJUGATE GRADIENT METHOD
#ak = backtracking line search
#xk+1 = xk + ak * pk
#pk = rk + beta * pk-1
#rk = b - A * xk
#beta = (rk^T * rk) / (rk-1^T * rk-1)


x1, x2, x3 = sp.symbols('x1 x2 x3')

#Error & Iteration values
e = 1e-5
itmax = 1000

# **2D (2 variables)**

#f'(x)
def derivative(f,x,y):
    dx1 = sp.diff(f, x1).subs([(x1,x), (x2,y)])
    dx2 = sp.diff(f, x2).subs([(x1,x), (x2,y)])

    M = sp.Matrix([dx1, dx2])
    return M

#steepest descent 2
def conjugate_grad_2(f,x,y,pk):
    G = derivative(f,x,y)
    rk = -G

    # pk = rk
    p0 = pk[0]
    p1 = pk[1]

    # Backtracking line search: shrink step size until function value decreases
    ak = 1.0
    f_current = float(f.subs({x1: x, x2: y}))
    while ak > 1e-12 and float(f.subs({x1: x + ak * p0, x2: y + ak * p1})) >= f_current:
        ak /= 2.0

    # # Update position: x_{k+1} = x_k + ak * pk
    # xk = xk + ak * pk
    x_new, y_new = x + ak * p0, y + ak * p1

    # # New residual: r_{k+1} = -f'(x_{k+1})
    rk_next = -derivative(f,x_new,y_new)

    # # Beta: (r_{k+1}^T * r_{k+1}) / (rk^T * rk)
    beta = (rk_next.T * rk_next)[0] / (rk.T * rk)[0]

    # # Update search direction: p_{k+1} = r_{k+1} + beta * pk
    p0 = rk_next[0] + beta * p0
    p1 = rk_next[1] + beta * p1
    pk = sp.Matrix([p0, p1])

    rk = rk_next

    return x_new, y_new, pk

    
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

#conjugate gradient 3
def conjugate_grad_3(f,x,y,z,pk):
    G = derivative_3(f,x,y,z)
    rk = -G
    ak = 1.0

    p0 = pk[0]
    p1 = pk[1]
    p2 = pk[2]

    f_current = float(f.subs({x1: x, x2: y, x3: z}))
    while ak > 1e-12 and float(f.subs({x1: x + ak * p0, x2: y + ak * p1, x3: z + ak * p2})) >= f_current:
        ak /= 2.0

    # # Update position: x_{k+1} = x_k + ak * pk
    # xk = xk + ak * pk
    x_new, y_new, z_new = x + ak * p0, y + ak * p1, z + ak * p2

    # # New residual: r_{k+1} = -f'(x_{k+1})
    rk_next = -derivative_3(f,x_new,y_new,z_new)

    # # Beta: (r_{k+1}^T * r_{k+1}) / (rk^T * rk)
    beta = (rk_next.T * rk_next)[0] / (rk.T * rk)[0]

    # # Update search direction: p_{k+1} = r_{k+1} + beta * pk
    p0 = rk_next[0] + beta * p0
    p1 = rk_next[1] + beta * p1
    p2 = rk_next[2] + beta * p2
    pk = sp.Matrix([p0, p1, p2])
    rk = rk_next

    return x_new, y_new, z_new, pk


def norm3(f,x,y,z):
    G = derivative_3(f,x,y,z)

    return math.sqrt(G[0]**2 + G[1]**2 + G[2]**2) / (1 + math.sqrt(f.subs([(x1,x), (x2,y), (x3,z)])**2))


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
    pk = -derivative_3(f,x,y,z)
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")
    try:
        while (it < itmax and norm3(f,x,y,z) > e):
            it +=1
            x,y,z,pk = conjugate_grad_3(f,x,y,z,pk)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")


#Looping through 2 var functions
for function in functions_2:
    f, (x, y) = function, functions_2[function]
    it = 0
    prob += 1
    pk = -derivative(f,x,y)
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    #print(f"problem {prob}:")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}\n")

    try:
        while (it < itmax and norm2(f,x,y) > e):
            it +=1
            x,y,pk = conjugate_grad_2(f,x,y,pk)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}")

#Variable function C = 1,10,100
prob += 1
for C in [1, 10, 100]:
    f, (x, y) = (x1-1)**2 + (x2-1)**2 + C*(x1**2 + x2**2 - 0.25)**2, (1.0,-1.0)
    it = 0
    pk = -derivative(f,x,y)
    print(f"-------------------------PROBLEM {prob}-------------------------\n")
    print(f"C = {C}\n")
    print(f"initial guess: x = {x:.4f}, y = {y:.4f}\n")
    print(f"function: {f}\n")

    try:
        while (it < itmax and norm2(f,x,y) > e):
            it +=1
            x,y,pk = conjugate_grad_2(f,x,y,pk)
            print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}\n")
    except ValueError as v:
        print(f"error: {v}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}")





    



