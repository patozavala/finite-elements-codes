# package are imported
import sympy as sp
import numpy as np


def get_residual_and_tangent(func):
    x, y = sp.symbols('x y')
    # residual calculation
    diff_x = sp.diff(func, x)
    diff_y = sp.diff(func, y)
    r = np.array([[diff_x, diff_y]]).T
    # tangent calculation
    diff_xx = sp.diff(diff_x, x)
    diff_yy = sp.diff(diff_y, y)
    diff_xy = sp.diff(diff_x, y)
    diff_yx = sp.diff(diff_y, x)
    t = np.array([[diff_xx, diff_yx], [diff_xy, diff_yy]])
    return r, t


def get_r_and_t(r_func, t_func, x_it):
    """
    this function gives the residual and tangent evaluated in x.
    Input:
    r_func: (sympy.core.mul.Mul) residual
    t_func: (sympy.core.mul.Mul) tangent
    x_it: point where the residual and the tangent will be evaluated.
    Ouput:
    r: residual evaluated in x
    t: residual evaluated in x
    """
    # the variables are initialized
    x, y = sp.symbols('x y')
    # the residual and tangent operator are calculated and evaluated in x_it ...
    r = np.array([[r_func[0, 0].subs([(x, x_it[0, 0]), (y, x_it[1, 0])]).evalf(),
                   r_func[1, 0].subs([(x, x_it[0, 0]), (y, x_it[1, 0])]).evalf()]]).astype(None)
    #    the tangent operator is calculated and evaluated in x_it ...
    t = np.array([[t_func[0, 0].subs([(x, x_it[0, 0]), (y, x_it[1, 0])]).evalf(),
                   t_func[0, 1].subs([(x, x_it[0, 0]), (y, x_it[1, 0])]).evalf()],
                  [t_func[1, 0].subs([(x, x_it[0, 0]), (y, x_it[1, 0])]).evalf(),
                   t_func[1, 1].subs([(x, x_it[0, 0]), (y, x_it[1, 0])]).evalf()]]).astype(None)
    return r, t


def newton_solver(func, solver_params):
    """
    this routine solves the Newton's method.
    :param func: function to solve
    :param solver_params: (list) parameters
    :return: None
    """
    # the variables are initialized
    x_it = solver_params['initial_point']
    error = 1E2
    nit = 0
    # the residual and tangent expressions are calculated
    r_func, t_func = get_residual_and_tangent(func)
    # start the iterations for the method...
    while nit < solver_params['maximum_iterations'] and error > solver_params['tolerance']:
        r, t = get_r_and_t(r_func, t_func, x_it)
        # the increment are computed...
        dx = -np.dot(np.linalg.inv(t), r.T)
        # the new point is calculated...
        x_it = x_it + dx
        # test for convergence
        error = np.linalg.norm(dx)
        print('residual norm : {}'.format(error))
        # if the number of iterations is exceeded then there is no convergence
        if nit == solver_params['maximum_iterations'] and error > solver_params['tolerance']:
            print('newton method does not converge')
        # if the error is less than the tolerance, then we have finished
        elif error < solver_params['tolerance']:
            print('newton method converge in {} iterations'.format(nit))
        else:
            nit = nit + 1


"""
Execution
"""

# some examples are defined (just to see results)
x, y = sp.symbols('x y')
f1 = sp.sin(x) * sp.sin(y)
f2 = (x - 1) ** 2 + (y - 1) ** 2

solver_parameters = {'tolerance': 1E-8,
                     'maximum_iterations': 50,
                     'initial_point': np.array([[10, 10]]).T}

newton_solver(f1, solver_params=solver_parameters)
newton_solver(f2, solver_params=solver_parameters)
