Illinois Institute of Technology
ECE 505 Project 1
Zubeda Patel A20532401

This repository contains 4 files appropriately names which implement the following optimization methods:
1. Steepest Descent Algorithm
2. Newton's Algorithm 
3. Quasi-Newton Algorithm (BFGS)
4. Conjugate Gradient Algorithm 

All of the above optimization implement the backtracking line search with Armijo's condition. 
It starts with ak = 1, then verfies armijo's condition :f(xk+akpk) > f(xk)+ c akpk^Tf'(xk). 
If True then ak = 1/2 ak otherwise otherwise ak = ak.
