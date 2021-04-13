# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 19:38:24 2017

@author: patricio
"""

"""
==============================
   NEWTON METHOD
==============================
"""
# The packages are imported...
import sympy as sp
import numpy as np


def getresidualandtangent(func):
    x, y = sp.symbols('x y')
#    the residual is calculated 
    derivate_x = sp.diff(func,x)
    derivate_y = sp.diff(func,y)
    Residualfunc= np.array([[derivate_x,derivate_y]]).T
#    the tangent operator is calculated 
    derivate_xx = sp.diff(derivate_x,x)
    derivate_yy = sp.diff(derivate_y,y)
    derivate_xy = sp.diff(derivate_x,y)
    derivate_yx = sp.diff(derivate_y,x)
    Tangentfunc= np.array([[derivate_xx,derivate_yx],[derivate_xy , derivate_yy]])
    return Residualfunc, Tangentfunc

def getRfunc(Residualfunc, Tangentfunc, x_it):
    """
    This function gives the residual and tangent operator evaluated in x. 
    
    Input:
        func        : function to derive. (depends on x)
        
        x_it           : point where the residual and the tangent operator will 
                      be evaluated.
        
    Ouput:
        Rfunc        : list with the residual and tangent operator evaluated in x
    """
#    the variables are inicialized...
    x, y = sp.symbols('x y')
#    the residual and tangent operator are calculated and evaluated in x_it ...
#    Residualfunc, Tangentfunc= getresidualandtangent(func) 
    Residual= np.array([[Residualfunc[0,0].subs([(x,x_it[0,0]),(y,x_it[1,0])]).evalf(), Residualfunc[1,0].subs([(x,x_it[0,0]),(y,x_it[1,0])]).evalf() ]]).astype(None)
#    the tangent operator is calculated and evaluated in x_it ...
    Tangent = np.array([[Tangentfunc[0,0].subs([(x,x_it[0,0]),(y,x_it[1,0])]).evalf(), Tangentfunc[0,1].subs([(x,x_it[0,0]),(y,x_it[1,0])]).evalf()], [Tangentfunc[1,0].subs([(x,x_it[0,0]),(y,x_it[1,0])]).evalf() , Tangentfunc[1,1].subs([(x,x_it[0,0]),(y,x_it[1,0])]).evalf()]]).astype(None)
    return Residual,Tangent

def NewtonSolver(func,solverparams,x0):
    """
    This function solves the Newton's method. 
    
    Input:
        x0           : initial point 
        Rfunc        : list with the residual and tangent operator evaluated in x
        solverparams : list with the necessary parameters to develop the Newton's 
                       method ... ([tol, nitmax])
                       where...  tol  --> tolerance
                                 nitmax --> number of iterations allowed
        
    Ouput:
        x            : 
        solverlog    : list with information about the iterations ([it,err,xit]), 
                       where...  it  --> number of iteration
                                 error --> euclidean norm of xit
                                 xit --> value of the solution in the iteration
    """
#    the variables are inicialized...
    tol = solverparams[0,0]
    nitmax = solverparams[0,1]
    x_it = x0
    error = tol +1 
    nit = 0
    ERROR=[]
    ERROR.append(error)
    xit=[] #the list which will be filled ...
    xit.append(x0)
#    the residual and tangent operator are calculated as a function...
    Residualfunc, Tangentfunc = getresidualandtangent(func)
#    start the iterations for the method...
    """ 
    G ----> Residual
    K ----> Tangent operator
    """
    while (nit < nitmax and error > tol):
        G , K = getRfunc(Residualfunc, Tangentfunc, x_it) 
#        the increment are computed...
        dx = -np.dot( np.linalg.inv(K), G.T)
#        the new point is calculated...
        x_it = x_it + dx
        xit.append(x_it)
#        test for convergence...
        error = np.linalg.norm(dx)
        ERROR.append(error)
#        if the number of iterations is exceeded then there is no convergence...
        if nit==nitmax and error>tol:
            print('no convergence')
#        if the error is less than the tolerance, then we have finished ....
        if error > tol:
            nit= nit + 1 
    return  nit, ERROR, xit