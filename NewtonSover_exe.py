# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 19:52:56 2017

@author: patricio
"""

"""
================================================
   ICE 3333 -  1ST HOMEWORK NON LINEAL - FEM 
================================================   
    This program is to execute the 1st problem in the 1st homework 
"""
import numpy as np
import sympy as sp
import matplotlib.pylab as plt
import NewtonSolver as Newton
from mpl_toolkits.mplot3d import Axes3D
import random

#The functions are defined ...
x, y = sp.symbols('x y')
##choose one function ... 
#func1 = sp.sin(x)*sp.sin(y)
func2 = (x - 1)**2 + (y - 1)**2
#func3 = sp.sqrt((x -1)**2 + (y -1)**2)


#The paremeters are setted ... 
tol= 10**(-8) #tolerance
nitmax = 100  #iterations
solverparams=np.array([[tol,nitmax]])
##choose one initial point ...
#x0 = np.array([[0,0]]).T 
#x0 = np.array([[1,1]]).T 
x0 = np.array([[10,10]]).T 

#the method is applied ...
nit, error , xit=Newton.NewtonSolver(func2,solverparams,x0)
#
#print('iterations',nit,'error', error ,'xit' ,xit)
# PLOTS AND FIGURES

##PLOT
#
#xit=np.asarray(xit)
#xit_x=np.zeros([xit.shape[0],1])
#xit_y=np.zeros([xit.shape[0],1])
#for i in np.arange(xit.shape[0]):
#    xit_x[i,0], xit_y[i,0] = xit[i,0,0] ,  xit[i,1,0]
#
## the coordinates to plot a contour line ...
#xplot= np.linspace(np.min(xit_x) - 1,np.max(xit_x)+1,50)
#yplot= np.linspace(np.min(xit_y) - 1,np.max(xit_y)+1,50)
##xplot= np.linspace(-10,10,50)
##yplot= np.linspace(-10,10,50)
#xplot,yplot= np.meshgrid(xplot,yplot)
## the function is computed in this points ...
#value = np.zeros([xplot.shape[0] , yplot.shape[0]])
#for j in np.arange(xplot.shape[0]):
#    for k in np.arange(yplot.shape[0]):
#        value[j,k]= func3.subs([(x,xplot[j,k]),(y,yplot[j,k])]).evalf()

##we create the figure()
#plt.figure()
##figure settings ... (remember to change the name of the function)
#plt.title('Función ' +'$\sqrt{(x -1)^{2} + (y -1)^{2}}$ ' + ' - Método Newton')
#plt.xlabel('x')
#plt.ylabel('y')
##we will plot the colormap and the points in each iteration ...
#plt.contourf(xplot,yplot, value ,  cmap=plt.cm.bone)
##for i in np.arange(xit_x.shape[0]-1):
##    plt.plot(xit_x[i],xit_y[i], 'bo' )
##    uncoment if you do not want to se the number of the iteration...
##    plt.text(x = xit_x[i], y = xit_y[i], s = i, fontsize = 20)
##plt.plot(xit_x,xit_y, 'bo')
#plt.colorbar()
#plt.savefig('newtonF3(2,2).png')
#plt.show()


# SUB-SECTION PLOT 

## the coordinates to plot a contour line ...
#xplot= np.linspace(np.min(xit_x) - 0.1,np.max(xit_x)+0.1,50)
#yplot= np.linspace(np.min(xit_y) - 0.1,np.max(xit_y)+0.1,50)
#
#xplot,yplot= np.meshgrid(xplot,yplot)
## the function is computed in this points ...
#value = np.zeros([xplot.shape[0] , yplot.shape[0]])
#for j in np.arange(xplot.shape[0]):
#    for k in np.arange(yplot.shape[0]):
#        value[j,k]= func2.subs([(x,xplot[j,k]),(y,yplot[j,k])]).evalf()
#
#
##we create the figure()
#plt.figure()
##figure settings ...
#plt.title('Función ' + '$(x -1)^{2} + (y -1)^{2}$ ' + ' - Método Newton')
#plt.xlabel('x')
#plt.ylabel('y')
##we will plot the colormap and the points in each iteration ...
#plt.contourf(xplot,yplot, value ,  cmap=plt.cm.bone)
#for i in np.arange(xit_x.shape[0]):
#    plt.plot(xit_x[i],xit_y[i], 'bo' )
##    uncoment if you do not want to se the number of the iteration...
##    plt.text(x = xit_x[i], y = xit_y[i], s = i, fontsize = 8)
#plt.colorbar()
#plt.savefig('newton-subsectionF2(0,0).png')
#plt.show()


#PLOT 3D 
#
#def fun(x, y):
#    return np.sin(x)*np.sin(y)
##  return (x -1)**2 + (y -1)**2
##
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#x = y = np.arange(0, 2*np.pi, 0.03)
#X, Y = np.meshgrid(x, y)
#zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
#Z = zs.reshape(X.shape)
#plt.title('Función ' +' $sin(x) sin (y)$ ' + ' 3D' )
#ax.plot_surface(X, Y, Z)
#ax.set_xlabel('X ')
#ax.set_ylabel('Y ')
#ax.set_zlabel('Z ')
#
#plt.show()

