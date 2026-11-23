import math
import sympy as sp

# Newton's method 
# Newton step: X_{k+1} = X_k - H^{-1}(X_k) * grad(f)(X_k)



x1, x2 = sp.symbols('x1 x2')
f = 5*x1**4 + 6*x2**4 - 6*x1**2 + 2*x1*x2 + 5*x2**2+15*x1-7*x2+13

a,b = 1.0, 1.0
e = 10e-5
itmax = 1000
it = 0



#f'(x)

def dx(a, b):
    return [
        20 * a**3 - 12 * a + 2 * b + 15,
        24 * b**3 + 2 * a + 10 * b - 7,
    ]

def derivative():
    dx1 = sp.diff(f, x1)
    dx2 = sp.diff(f, x2)
    return dx1, dx2 

# f''(x)
def second_derivative():
    
    dx1, dx2 = derivative()

    dx1x1 = sp.diff(dx1, x1)
    dx1x2 = sp.diff(dx1, x2)
    dx2x1 = sp.diff(dx2, x1)
    dx2x2 = sp.diff(dx2, x2)
    return dx1x1, dx1x2, dx2x1, dx2x2


def dxx(a, b):
    return [
        [60 * a**2 - 12, 2],
        [2, 72 * b**2 + 10],
    ]

# Solve H * s = g for s, then update x = x - s
# This is the 2D equivalent of x_{k+1} = x_k - f'(x)/f''(x)
def newton_step_2():
    #g depends on # of variables
    g1,g2 = derivative()
    g1 = g1.subs([(x1,1), (x2,1)])
    g2 = g2.subs([(x1,1), (x2,1)])
    
    H1, H2, H3, H4 = second_derivative()
    H1 = H1.subs([(x1,1), (x2,1)])
    H2 = H2.subs([(x1,1), (x2,1)])
    H3 = H3.subs([(x1,1), (x2,1)])
    H4 = H4.subs([(x1,1), (x2,1)])


    det = H1 * H4 - H2 * H3
    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")
    s1 = (H4 * g1 - H2 * g2) / det
    s2 = (-H3 * g1 + H1 * g2) / det
    return x1 - s1, x1 - s2 #WORKS BUT GOTTA MAKE IT SPECIFICS


def newton_step(a, b):
    g = dx(a, b)
    H = dxx(a, b)

    a1, a2 = H[0]
    b1, b2 = H[1]
    det = a1 * b2 - a2 * b1

    if abs(det) < 1e-12:
        raise ValueError("Hessian is singular; Newton step fails.")

    # Solve:

    s1 = (b2 * g[0] - a2 * g[1]) / det
    s2 = (-b1 * g[0] + a1 * g[1]) / det

    return (a-s1), (b-s2)

#iteration
#calculate ||dx|| / (1+||ddxy||)
def comp(a,b):
    g1,g2 = derivative()
    g1 = g1.subs([(x1,a), (x2,b)])
    g2 = g2.subs([(x1,a), (x2,b)])

    H1, H2, H3, H4 = second_derivative()

    H1 = H1.subs([(x1,a), (x2,b)])
    H2 = H2.subs([(x1,a), (x2,b)])
    H3 = H3.subs([(x1,a), (x2,b)])
    H4 = H4.subs([(x1,a), (x2,b)])
    det = H1 * H4 - H2 * H3
    
    s1 = (H4 * g1 - H2 * g2) / det
    s2 = (-H3 * g1 + H1 * g2) / det
    return math.sqrt(s1**2 + s2**2) / (1 + math.sqrt(g1**2 + g2**2))


print("general:", derivative())
print("actual:", dx(a,b))
print("general:", second_derivative())
print("actual:", dxx(a, b))
print("general:", newton_step_2())
print("actual:", newton_step(a, b))

#x0 = (0,0)
x = 0
y = 0
while (it < itmax and comp(x,y) > e):
    it +=1
    x,y = newton_step(x,y)
    print(f"iteration {it}: x = {x:.4f}, y = {y:.4f}\n")

print(f"final: x = {x:.4f}, y = {y:.4f}\n")







    


