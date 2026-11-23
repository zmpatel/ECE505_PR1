import math
import sympy as sp

# Newton's method 
# Newton step: X_{k+1} = X_k - H^{-1}(X_k) * grad(f)(X_k)

x1, x2, x3 = sp.symbols('x1 x2 x3')


#Error & Iteration values
e = 10e-5
itmax = 1000

#2D (2 variables)

#f'(x)
def derivative(f):
    dx1 = sp.diff(f, x1)
    dx2 = sp.diff(f, x2)
    return dx1, dx2 

# f''(x)
def second_derivative(f):
    
    dx1, dx2 = derivative(f)

    dx1x1 = sp.diff(dx1, x1)
    dx1x2 = sp.diff(dx1, x2)
    dx2x1 = sp.diff(dx2, x1)
    dx2x2 = sp.diff(dx2, x2)
    return dx1x1, dx1x2, dx2x1, dx2x2

# Solve H * s = g for s, then update x = x - s
# This is the 2D equivalent of x_{k+1} = x_k - f'(x)/f''(x)
def newton_step_2(f,x,y):
    #g depends on # of variables
    g1,g2 = derivative(f)
    g1 = g1.subs([(x1,x), (x2,y)])
    g2 = g2.subs([(x1,x), (x2,y)])
    
    H1, H2, H3, H4 = second_derivative(f)
    H1 = H1.subs([(x1,x), (x2,y)])
    H2 = H2.subs([(x1,x), (x2,y)])
    H3 = H3.subs([(x1,x), (x2,y)])
    H4 = H4.subs([(x1,x), (x2,y)])


    det = H1 * H4 - H2 * H3
    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")
    s1 = (H4 * g1 - H2 * g2) / det
    s2 = (-H3 * g1 + H1 * g2) / det

    

    print(f"search direction: [{s1:.4f}, {s2:.4f}]")
    print(f"step length: ||s|| = {math.sqrt(s1**2 + s2**2):.4f}\n")

    return x - s1, y - s2 #WORKS BUT GOTTA MAKE IT SPECIFICS

#iteration
#calculate ||dx|| / (1+||ddxy||)
def comp2(f,a,b):
    g1,g2 = derivative(f)
    g1 = g1.subs([(x1,a), (x2,b)])
    g2 = g2.subs([(x1,a), (x2,b)])

    H1, H2, H3, H4 = second_derivative(f)

    H1 = H1.subs([(x1,a), (x2,b)])
    H2 = H2.subs([(x1,a), (x2,b)])
    H3 = H3.subs([(x1,a), (x2,b)])
    H4 = H4.subs([(x1,a), (x2,b)])
    det = H1 * H4 - H2 * H3

    # if abs(det) < 1e-12:
    #     raise ValueError("Hessian is singular; Newton step fails.")
    
    s1 = (H4 * g1 - H2 * g2) / det
    s2 = (-H3 * g1 + H1 * g2) / det
    return math.sqrt(s1**2 + s2**2) / (1 + math.sqrt(g1**2 + g2**2))

#3D (3 variables)

#f'(x)
def derivative_3(f):
    dx1 = sp.diff(f, x1)
    dx2 = sp.diff(f, x2)
    dx3 = sp.diff(f, x3)
    return dx1, dx2, dx3

#f''(x)
def second_derivative_3(f):
    dx1, dx2, dx3 = derivative_3(f)

    dx1x1 = sp.diff(dx1, x1)
    dx1x2 = sp.diff(dx1, x2)
    dx1x3 = sp.diff(dx1, x3)
    dx2x1 = sp.diff(dx2, x1)
    dx2x2 = sp.diff(dx2, x2)
    dx2x3 = sp.diff(dx2, x3)
    dx3x1 = sp.diff(dx3, x1)
    dx3x2 = sp.diff(dx3, x2)
    dx3x3 = sp.diff(dx3, x3)
    return dx1x1, dx1x2, dx1x3, dx2x1, dx2x2, dx2x3, dx3x1, dx3x2, dx3x3

#newtons step 3
def newton_step_3(f,x,y,z):
    g1,g2,g3 = derivative_3(f)
    g1 = g1.subs([(x1,x), (x2,y), (x3,z)])
    g2 = g2.subs([(x1,x), (x2,y), (x3,z)])
    g3 = g3.subs([(x1,x), (x2,y), (x3,z)])

    H1, H2, H3, H4, H5, H6, H7, H8, H9 = second_derivative_3(f)
    H1 = H1.subs([(x1,x), (x2,y), (x3,z)])
    H2 = H2.subs([(x1,x), (x2,y), (x3,z)])
    H3 = H3.subs([(x1,x), (x2,y), (x3,z)])
    H4 = H4.subs([(x1,x), (x2,y), (x3,z)])
    H5 = H5.subs([(x1,x), (x2,y), (x3,z)])
    H6 = H6.subs([(x1,x), (x2,y), (x3,z)])
    H7 = H7.subs([(x1,x), (x2,y), (x3,z)])
    H8 = H8.subs([(x1,x), (x2,y), (x3,z)])
    H9 = H9.subs([(x1,x), (x2,y), (x3,z)])

    det = H1*(H5*H9 - H6*H8) - H2*(H4*H9 - H6*H7) + H3*(H4*H8 - H5*H7)
    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")
    s1 = ((H5*H9 - H6*H8)*g1 - (H2*H9 - H3*H8)*g2 + (H2*H6 - H3*H5)*g3) / det
    s2 = (-(H4*H9 - H6*H7)*g1 + (H1*H9 - H3*H7)*g2 - (H1*H6 - H3*H4)*g3) / det
    s3 = ((H4*H8 - H5*H7)*g1 - (H1*H8 - H2*H7)*g2 + (H1*H5 - H2*H4)*g3) / det

    print(f"search direction: [{s1:.4f}, {s2:.4f},{s3:.4f}]")
    print(f"step length: ||s|| = {math.sqrt(s1**2 + s2**2 + s3**2):.4f}\n")

    return x - s1, y - s2, z - s3

def comp3(f,a,b,c):
    g1,g2,g3 = derivative_3(f)
    g1 = g1.subs([(x1,a), (x2,b), (x3,c)])
    g2 = g2.subs([(x1,a), (x2,b), (x3,c)])
    g3 = g3.subs([(x1,a), (x2,b), (x3,c)])
    H1, H2, H3, H4, H5, H6, H7, H8, H9 = second_derivative_3(f)
    H1 = H1.subs([(x1,a), (x2,b), (x3,c)])
    H2 = H2.subs([(x1,a), (x2,b), (x3,c)])
    H3 = H3.subs([(x1,a), (x2,b), (x3,c)])
    H4 = H4.subs([(x1,a), (x2,b), (x3,c)])
    H5 = H5.subs([(x1,a), (x2,b), (x3,c)])
    H6 = H6.subs([(x1,a), (x2,b), (x3,c)])
    H7 = H7.subs([(x1,a), (x2,b), (x3,c)])
    H8 = H8.subs([(x1,a), (x2,b), (x3,c)])
    H9 = H9.subs([(x1,a), (x2,b), (x3,c)])

    M = sp.Matrix([[H1, H2, H3], [H4, H5, H6], [H7, H8, H9]])
    det = M.det()

    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")

    s1, s2, s3 = M.LUsolve(sp.Matrix([g1, g2, g3]))
    return math.sqrt(s1**2 + s2**2 + s3**2) / (1 + math.sqrt(g1**2 + g2**2 + g3**2))



#AAA @each iteration print search direction, step length, AAA
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

functions_3 = {
                x1**2 + x2**2 + x3**2 : (1.0,1.0,1.0)
}
prob = 0;
for function in functions_2:
    f, (x, y) = function, functions_2[function]
    it = 0
    prob += 1
    print(f"problem {prob}:")
    while (it < itmax and comp2(f,x,y) > e):
        it +=1
        x,y = newton_step_2(f,x,y)
        print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}\n")

for function in functions_3:
    f, (x, y, z) = function, functions_3[function]
    it = 0
    prob += 1
    print(f"problem {prob}:")
    while (it < itmax and comp3(f,x,y,z) > e):
        it +=1
        x,y,z = newton_step_3(f,x,y,z)
        print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")

    print(f"final: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}\n")







    


