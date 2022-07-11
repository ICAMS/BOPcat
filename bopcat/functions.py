#!/usr/bin/env python

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import numpy as np
from scipy import optimize
from copy import deepcopy
from .output import print_format
from math import *

"""
This module contains all the functional forms for fitting.

Part of the BOPcat package.

author: Alvin Noe Ladines
e-mail: ladinesalvinnoe@gmail.com
"""


class Function:
    """
    Generic class for a function.

    :Parameters:
    
        - *parameters*: list

            list of floats that can be set in the function

        - *numbers*: list

            list of floats that contain all the parameters 
            and constants in the funtion

            entry can also be a string corresponding to a
            python operation that is executed in the function
            e.g. [1,'numbers[0]*2+exp(numbers[2])',2]
            operation should be in math library

        - *constraints*: list
 
            list of logical flags to set an entry in number a
            free parameter

        - *name*: name of the function
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1]
        self._constraints = [True, True, False]
        self._numbers = [1, 1, 1]
        self._name = 'exponential'
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key.lower() == 'parameters':
                self._parameters = val
            if key.lower() == 'constraints':
                self._constraints = val
            if key.lower() == 'numbers':
                self._numbers = val
            if key.lower() == 'name':
                self._name = val
        if len(self._constraints) != len(self._numbers):
            print(('_constraints = ', self._constraints))
            print(('_numbers = ', self._numbers))
            msg = 'Length of constraints should be %d' % len(self._numbers)
            raise ValueError(msg)
        if len(self._parameters) != self._constraints.count(True):
            msg = 'Mismatch between number of parameters and constraints.'
            raise ValueError(msg)

    def set_parameters(self, p):
        self._parameters = list(p)
        count = 0
        for i in range(len(self._constraints)):
            if self._constraints[i]:
                self._numbers[i] = self._parameters[count]
                count += 1

    def get_numbers(self):
        self.set_parameters(self._parameters)
        old_numbers = list(self._numbers)
        for i in range(len(self._numbers)):
            if isinstance(self._numbers[i], str):
                assert (self._constraints[i] == False)
                exec("self._numbers[%d] = %s" % (i, 'self._numbers'.join( \
                    self._numbers[i].split('numbers'))))
        new_numbers = list(self._numbers)
        self._numbers = list(old_numbers)
        return new_numbers

    def get_constraints(self):
        return list(self._constraints)

    def get_parameters(self):
        return list(self._parameters)

    def copy(self):
        return deepcopy(self)


###########################################################################
class exponential(Function):
    """
    Simple exponential function
    y = A * exp(-B*R)  
  
    maximum number of parameters : 2
    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1]
        self._constraints = [True, True]
        self._numbers = [1, 1]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        y = A * np.exp(-B * x)
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "exponential"
            elif environment == 'bopfox-ham':
                name = "exponential"
            elif environment == 'bopfox-rep':
                name = 'pp_exponential'
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        s = '%s*exp(-%s*(X))' % (A, B)
        return s


###########################################################################
class exponential_e(Function):
    """
    Simple exponential function with cosine cut off
    y = A * exp(-B*R)  
  
    maximum number of parameters : 
    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1, 1]
        self._constraints = [True, True, True]
        self._numbers = [1, 1, 1]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        y = A * np.exp(-B * (x ** C))
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "exponential_e"
            elif environment == 'bopfox-ham':
                name = "exponential_e"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class screenedexponential(Function):
    """
    screened exponential function
    
    y = A*exp( -B*(x/E-1.) )*( 1. - exp( -C*(x/E)**D ) )
  
    maximum number of parameters : 5
    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1, 1, 1, 1]
        self._constraints = [True, True, True, True, True]
        self._numbers = [1, 1, 1, 1, 1]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        E = numbers[4]
        y = A * np.exp(-B * (x / E - 1.)) * (1. - np.exp(-C * (x / E) ** D))
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "screenedexponential"
            elif environment == 'bopfox-ham':
                name = "screenedexponential"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class shiftedexponential(Function):
    """
    screened exponential function
    
    y = A *exp(-B*(x/C-1.))
  
    maximum number of parameters : 5
    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1, 1]
        self._constraints = [True, True, True]
        self._numbers = [1, 1, 1]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        y = A * np.exp(-B * (x / C - 1.))
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "shiftedexponential"
            elif environment == 'bopfox-ham':
                name = "shiftedexponential"
            elif environment == 'bopfox-rep':
                name = "pp_shiftedexponential"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        s = '%s*exp(-%s*(X/%s-1.))' % (A, B, C)
        return s


###########################################################################
class exponential7(Function):
    """
    sum of seven exponential functions
    y = sum_i A_i * exp(-B_i*R^C_i)

    maximum number of parameters : 21
    Note: Future functions should at least have the
            same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1, 1] * 7
        self._constraints = [True, True, True] * 7
        self._numbers = [1, 1, 1] * 7
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        y = np.zeros(len(x))
        for i in range(7):
            A = numbers[i * 3]
            B = numbers[i * 3 + 1]
            C = numbers[i * 3 + 2]
            y += A * np.exp(-B * (x ** C))
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "exponential7"
            elif environment == 'bopfox-ham':
                name = "sum7exp"
            elif environment == 'bopfox-rep':
                name = "pp_sum7exp"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        s = ''
        for i in range(7):
            A = numbers[i * 7]
            B = numbers[i * 7 + 1]
            C = numbers[i * 7 + 2]
            s += '+%s*exp(-%s*(X**%s))' % (A, B, C)
        return s


###########################################################################
class powerlaw(Function):
    """
    power law
    
    y = A *(x/C)**B

    maximum number of parameters : 4 (D not used)
    Note: Future functions should at least have the
            same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1, 1]
        self._constraints = [True, True, True, False]
        self._numbers = [1, 1, 1, 0]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        y = A * (x / C) ** B
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "powerlaw"
            elif environment == 'bopfox-ham':
                name = "powerlaw"
            elif environment == 'bopfox-rep':
                name = "pp_powerlaw"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        s = '%s *(X/%s)**%s' % (A, C, B)
        return s


###########################################################################
class screeenedpowerlaw(Function):
    """
    screened power law
    exp1 = (B/C)**F
    Y = A*(x/B)**(-D) *exp(E*(exp1-(x/C)**F))

    maximum number of parameters = 7 (G not used)

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1, 1, 1, 1, 1, 1]
        self._constraints = [True, True, True, True, True, True, True]
        self._numbers = [1, 1, 1, 1, 1, 1, 1]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        E = numbers[4]
        F = numbers[5]
        G = numbers[6]
        exp1 = (B / C) ** F
        y = A * (x / B) ** (-D) * np.exp(E * (exp1 - (x / C) ** F))
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "gsp"
            elif environment == 'bopfox-ham':
                name = "screenedpowerlaw"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        E = numbers[4]
        F = numbers[5]
        G = numbers[6]
        exp1 = (B / C) ** F
        s = '%s*(X/%s)**(-%s) *exp(%s*(%s-(X/%s)**%s))' % (A, B, D, E, exp1, C, F)
        return s


###########################################################################
class screeenedpowerlaw_polynom5(Function):
    """
    screened power law with polynomial cut off
    exp1 = (B/C)**F
    Y = A*(x/B)**(-D) *exp(E*(exp1-(x/C)**F))

    maximum number of parameters = 8 

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1, 1, 1, 1, 1, 1, 1]
        self._constraints = [True, True, True, True, True, True, True, True]
        self._numbers = [1, 1, 1, 1, 1, 1, 1, 1]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        E = numbers[4]
        F = numbers[5]
        G = numbers[6]
        H = numbers[7]
        exp1 = (B / C) ** F
        y = A * (x / B) ** (-D) * np.exp(E * (exp1 - (x / C) ** F))
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "gsp"
            elif environment == 'bopfox-ham':
                name = "screenedpowerlaw_polynom5"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError

    ###########################################################################


class powerfraction(Function):
    """
    power fraction
    
    y = B*(A-x)**C / x**D

    maximum number of parameters : 4 
    Note: Future functions should at least have the
            same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1, 1]
        self._constraints = [True, True, True, False]
        self._numbers = [1, 1, 1, 0]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        y = B * (A - x) ** C / x ** D
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "powerfraction"
            elif environment == 'bopfox-ham':
                name = "powerfraction"
            elif environment == 'bopfox-rep':
                name = "pp_powerfraction"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        s = '%s*(%s-X)**%s / X**%s' % (B, A, C, D)
        return s

    ###########################################################################


class screenedpowerfraction(Function):
    """
    screened power fraction
    
    y = B*(A-x)**3./x*exp(A-x)

    maximum number of parameters : 2 
    Note: Future functions should at least have the
            same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1]
        self._constraints = [True, True]
        self._numbers = [1, 1]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        y = B * (A - x) ** 3. / x * np.exp(A - x)
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "powerfraction"
            elif environment == 'bopfox-ham':
                name = "screenedpowerfraction"
            elif environment == 'bopfox-rep':
                name = "pp_screenedpowerfraction"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        s = '%s*(%s-X)**3./X*np.exp(%s-X)' % (B, A, A)
        return s

    ###########################################################################


class Yukawa(Function):
    """
    Yukawa function
    
    y = A *exp(-B*x) /x

    maximum number of parameters : 2 
    Note: Future functions should at least have the
            same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1]
        self._constraints = [True, True]
        self._numbers = [1, 1]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        y = A * np.exp(-B * x) / x
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "yukawa"
            elif environment == 'bopfox-ham':
                name = "yukawa"
            elif environment == 'bopfox-rep':
                name = "pp_yukawa"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        s = '%s*exp(-%s*X) /X' % (A, B)
        return s

    ###########################################################################


class oscillator(Function):
    """
    polynomial function    
    
    y = A + B*(x-C)**2. + D*x
    
    maximum number of parameters = 4 

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1, 1, 1]
        self._numbers = [1, 1, 1, 1]
        self._constraints = [True, True, True, True]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        y = A + B * (x - C) ** 2. + D * x
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "oscillator"
            elif environment == 'bopfox-ham':
                name = "osc"
            elif environment == 'bopfox-rep':
                name = "pp_osc"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        s = '%s + %s*(X-%s)**2. + %s*X' % (A, B, C, D)
        return s


###########################################################################
class spline4(Function):
    """
    four splines    
    
    maximum number of parameters = 8 

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [-0.3, 1.4, -2.9, 7.3, 4.5, 4.0, 3.3, 3.1]
        self._numbers = [-0.3, 1.4, -2.9, 7.3, 4.5, 4.0, 3.3, 3.1]
        self._constraints = [True, True, True, True, True, True, True, True]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0:4]
        R = numbers[4:8]
        y = np.zeros_like(x)
        for i in range(len(x)):
            if x[i] < R[3]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3 \
                       + A[2] * (R[2] - x[i]) ** 3 + A[3] * (R[3] - x[i]) ** 3
            elif x[i] < R[2]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3 \
                       + A[2] * (R[2] - x[i]) ** 3
            elif x[i] < R[1]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3
            elif x[i] < R[0]:
                y[i] = A[0] * (R[0] - x[i]) ** 3
            else:
                y[i] = 0.
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "spline4"
            elif environment == 'bopfox-ham':
                name = "spline4"
            elif environment == 'bopfox-rep':
                name = "pp_spline4"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        return NotImplementedError


###########################################################################
class spline5(Function):
    """
    five splines    
    
    maximum number of parameters = 10

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [-0.67, 1.12, -0.87, 1.77, 10.0, 4.2, 4.0, 3.1
            , 3.0, 2.37]
        self._numbers = [-0.67, 1.12, -0.87, 1.77, 10.0, 4.2, 4.0, 3.1, 3.0
            , 2.37]
        self._constraints = [True, True, True, True, True, True, True, True, True, True]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0:5]
        R = numbers[5:10]
        y = np.zeros_like(x)
        for i in range(len(x)):
            if x[i] < R[4]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3 \
                       + A[2] * (R[2] - x[i]) ** 3 + A[3] * (R[3] - x[i]) ** 3 \
                       + A[4] * (R[4] - x[i]) ** 3
            elif x[i] < R[3]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3 \
                       + A[2] * (R[2] - x[i]) ** 3 + A[3] * (R[3] - x[i]) ** 3
            elif x[i] < R[2]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3 \
                       + A[2] * (R[2] - x[i]) ** 3
            elif x[i] < R[1]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3
            elif x[i] < R[0]:
                y[i] = A[0] * (R[0] - x[i]) ** 3
            else:
                y[i] = 0.
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "spline5"
            elif environment == 'bopfox-ham':
                name = "spline5"
            elif environment == 'bopfox-rep':
                name = "pp_spline5"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        return NotImplementedError


###########################################################################
class spline6(Function):
    """
    six splines    
    
    maximum number of parameters = 12

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [5.0, 5.0, 4.6, 3.9, 3.3, 2.9, -0.7, 1.1, -0.2, -0.1, 1.0, 0.4]
        self._numbers = [5.0, 5.0, 4.6, 3.9, 3.3, 2.9, -0.7, 1.1, -0.2, -0.1, 1.0, 0.4]
        self._constraints = [True] * 12
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0:5]
        R = numbers[5:10]
        y = np.zeros_like(x)
        for i in range(len(x)):
            if x[i] < R[5]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3 \
                       + A[2] * (R[2] - x[i]) ** 3 + A[3] * (R[3] - x[i]) ** 3 \
                       + A[4] * (R[4] - x) ** 3 + A[5] * (R[5] - x) ** 3
            elif x[i] < R[4]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3 \
                       + A[2] * (R[2] - x[i]) ** 3 + A[3] * (R[3] - x[i]) ** 3 \
                       + A[4] * (R[4] - x) ** 3
            elif x[i] < R[3]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3 \
                       + A[2] * (R[2] - x[i]) ** 3 + A[3] * (R[3] - x[i]) ** 3
            elif x[i] < R[2]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3 \
                       + A[2] * (R[2] - x[i]) ** 3
            elif x[i] < R[1]:
                y[i] = A[0] * (R[0] - x[i]) ** 3 + A[1] * (R[1] - x[i]) ** 3
            elif x[i] < R[0]:
                y[i] = A[0] * (R[0] - x[i]) ** 3
            else:
                y[i] = 0.
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "spline6"
            elif environment == 'bopfox-ham':
                name = "spline6"
            elif environment == 'bopfox-rep':
                name = "pp_spline6"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        return NotImplementedError


###########################################################################
class polynomial4(Function):
    """
    four polynomial functions    
    
    y = (x-A)**2 * (B + C*x + D*x**2)
    
    maximum number of parameters = 4

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1, 1, 1]
        self._numbers = [1, 1, 1, 1]
        self._constraints = [True, True, True, True]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        y = (x - A) ** 2 * (B + C * x + D * x ** 2)
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "polynomial4"
            elif environment == 'bopfox-ham':
                name = "polynom4"
            elif environment == 'bopfox-rep':
                name = "pp_polynom4"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        s = '(X-%s)**2 * (%s + %s*X + %s*X**2)' % (A, B, C, D)
        return s


###########################################################################
class polynomial(Function):
    """
    polynomial function    
    
    y = A * (x - B) ^ C
    
    maximum number of parameters = 3 

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1]
        self._numbers = [1, 0, 3]
        self._constraints = [True, False, False]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        y = A * ((x - B) ** C)
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "polynomial"
            elif environment == 'bopfox-ham':
                name = "polynomial"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        s = '%s*((X-%s)**%s)' % (A, B, C)
        return s


###########################################################################
class root_gaussian7(Function):
    """ 
    rho = sum_i(7) A_i *exp(-B_i*(x-C_i)**2 )
    
    f = (rho)**N
    
    g = exp(-K*abs(f))

    y = sign(rho)*(1-g)*f + g*rho
    
    maximum number of parameters = 23

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [-1, 1]
        self._numbers = [-1, 1, 0] + [0, 0, 0] * 6 + [0.5, 1.0e5]
        self._constraints = [True, True, False] + [False, False, False] * 6 + [False] * 2
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        rho = np.zeros_like(x)
        for i in range(7):
            A = numbers[0 + (i * 3)]
            B = numbers[1 + (i * 3)]
            C = numbers[2 + (i * 3)]
            rho += A * np.exp(-B * (x - C) ** 2)
        y = rho
        # f = abs(rho)**numbers[21]
        ##g = np.exp(-numbers[22]*np.abs(f))
        # g = np.exp(-numbers[22]*np.abs(rho))
        # y = np.sign(rho)*(1.0-g)*f + g*rho
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "root_gaussian7"
            elif environment == 'bopfox-ham':
                name = "root_sum7gaussian"
            elif environment == 'bopfox-rep':
                name = "emb_root_sum7gaussian"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class gaussian(Function):
    """ 
    rho = A *exp(-B*(x-C)**2 )
    
    y = (rho)**N
    
    maximum number of parameters = 4

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [-1, 1, 1]
        self._numbers = [-1, 1, 1, 0.5]
        self._constraints = [True, True, True, False]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        rho = np.zeros_like(x)
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        rho += A * np.exp(-B * (x - C) ** 2)
        y = rho
        # f = abs(rho)**numbers[21]
        ##g = np.exp(-numbers[22]*np.abs(f))
        # g = np.exp(-numbers[22]*np.abs(rho))
        # y = np.sign(rho)*(1.0-g)*f + g*rho
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "gaussian"
            elif environment == 'bopfox-ham':
                name = "gaussian"
            elif environment == 'bopfox-rep':
                name = "emb_sqrt_gaussian"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################

class sqrt_gaussian(Function):
    """ 
    
    y = (A *exp(-B*x**2 ))**0.5
    
    maximum number of parameters = 2 

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1, 1]
        self._numbers = [1, 1]
        self._constraints = [True, True]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        y = (A ** 2 * np.exp(-B * x ** 2)) ** 0.5
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "sqrt_gaussian"
            elif environment == 'bopfox-ham':
                name = "sqrt_gaussian"
            elif environment == 'bopfox-rep':
                name = "emb_sqrt_gaussian"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        s = '(%s**2*exp(-%s*X**2 ))**0.5' % (A, B)
        return s


###########################################################################
class powerlaw_exp(Function):
    """    
    
    rhoij = B * exp(-C*(rij/A-1.d0))

    y  = F * rhoij**G
    
    maximum number of parameters = 7

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [2.7, 3.4, 5.6, 0., 0., -1.0, 1.0]
        self._numbers = [2.7, 3.4, 5.6, 0., 0., -1.0, 1.0]
        self._constraints = [True, True, True, True, True, True, True]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        D = numbers[3]
        E = numbers[4]
        F = numbers[5]
        G = numbers[6]
        rhoij = B * np.exp(-C * (x / A - 1.0))
        y = F * rhoij ** G
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "powerlaw_exp"
            elif environment == 'bopfox-ham':
                name = "powerlaw_exp"
            elif environment == 'bopfox-rep':
                name = "emb_powerlaw_exp"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class sqrt_polynom3(Function):
    """    
    
    rhoij = C * ((x-A)**2 + B/A*(x-A)**3.)

    y  = rhoij**0.5
    
    maximum number of parameters = 3

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [3.7, 2.5, 5.9]
        self._numbers = [3.7, 2.5, 5.9]
        self._constraints = [True, True, True]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        A = numbers[0]
        B = numbers[1]
        C = numbers[2]
        rhoij = C * ((x - A) ** 2 + B / A * (x - A) ** 3.)
        y = rhoij ** 0.5
        return y

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "sqrt_polynom3"
            elif environment == 'bopfox-ham':
                name = "sqrt_polynom3"
            elif environment == 'bopfox-rep':
                name = "emb_sqrt_polynom3"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class sqrt_spline2(Function):
    """    
    maximum number of parameters = 4

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [5.0, 4.3, 0.5, -0.5]
        self._numbers = [5.0, 4.3, 0.5, -0.5]
        self._constraints = [True] * 4
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        raise NotImplementedError

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "sqrt_spline2"
            elif environment == 'bopfox-ham':
                name = "sqrt_spline2"
            elif environment == 'bopfox-rep':
                name = "emb_sqrt_spline2"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class powerlaw_gsp_hc(Function):
    """    
    maximum number of parameters = 21

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [20.8, 2.3, 3.1, 4.6, 4.6, 2.3, 1.0, 0.5, 4.1, 4.3, 0.2]
        self._numbers = [20.8, 2.3, 3.1, 4.6, 4.6, 2.3, 1.0, 0.5, 4.1, 4.3, 0.2,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._constraints = [True] * 11 + [False] * 10
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        raise NotImplementedError

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "powerlaw_gsp_hc"
            elif environment == 'bopfox-ham':
                name = "powerlaw_gsp_hc"
            elif environment == 'bopfox-rep':
                name = "emb_powerlaw_gsp_hc"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class powerlaw_gsp(Function):
    """    
    maximum number of parameters = 11

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1.0, 1.5, 4.3, 9.3, 2.2, 0, 7.7, 0.8, 2.5, 2.6, -0.7]
        self._numbers = [1.0, 1.5, 4.3, 9.3, 2.2, 0, 7.7, 0.8, 2.5, 2.6, -0.7]
        self._constraints = [True] * 11
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        raise NotImplementedError

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "powerlaw_gsp"
            elif environment == 'bopfox-ham':
                name = "powerlaw_gsp"
            elif environment == 'bopfox-rep':
                name = "emb_powerlaw_gsp"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class polynom4_gsp(Function):
    """    
    maximum number of parameters = 12

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [7.2, 1.5, 3.6, 9.2, 2.3, 1.0, 0.0, 0.0, 0.0, 2.3, 2.60, 0.0]
        self._numbers = [7.2, 1.5, 3.6, 9.2, 2.3, 1.0, 0.0, 0.0, 0.0, 2.3, 2.60, 0.0]
        self._constraints = [True] * 12
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        raise NotImplementedError

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "polynom4_gsp"
            elif environment == 'bopfox-ham':
                name = "polynom4_gsp"
            elif environment == 'bopfox-rep':
                name = "emb_polynom4_gsp"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class env_yukawa(Function):
    """    
    maximum number of parameters = 8

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [22.0, 1.0, 2.0, 110.0, 2.0, 1.5, 0.0, 5.0]
        self._numbers = [22.0, 1.0, 2.0, 110.0, 2.0, 1.5, 0.0, 5.0]
        self._constraints = [True] * 8
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        raise NotImplementedError

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "env_yukawa"
            elif environment == 'bopfox-ham':
                name = "env_yukawa"
            elif environment == 'bopfox-rep':
                name = "env_yukawa"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class env_yukawa_polynom(Function):
    """    
    maximum number of parameters = 9

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [37.21, 1.0, 2.0, 110.0, 1.5, 2.0, 3.1, 4.2, 1.0]
        self._numbers = [37.21, 1.0, 2.0, 110.0, 1.5, 2.0, 3.1, 4.2, 1.0]
        self._constraints = [True] * 9
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        raise NotImplementedError

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "env_yukawa_polynom"
            elif environment == 'bopfox-ham':
                name = "env_yukawa_polynom"
            elif environment == 'bopfox-rep':
                name = "env_yukawa_polynom"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class env_closedshell(Function):
    """    
    maximum number of parameters = 9

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [16330, 13.0, 0.5, 100.0, 10.0, 1.0, 1.0, 1.3, 3.0]
        self._numbers = [16330, 13.0, 0.5, 100.0, 10.0, 1.0, 1.0, 1.3, 3.0]
        self._constraints = [True] * 9
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        raise NotImplementedError

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "env_closedshell"
            elif environment == 'bopfox-ham':
                name = "env_closedshell"
            elif environment == 'bopfox-rep':
                name = "env_closeshell"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class reduced_tb(Function):
    """    
    maximum number of parameters = 14

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [6.7, 1.4, 7.3, 1.0, -9.6, 1.7, 2.5, 1.1, 1.1, 3.0, 2.7, 4.0]
        self._numbers = [1.0, 1, 6.7, 1.4, 7.3, 1.0, -9.6, 1.7, 2.5, 1.1, 1.1, 3.0, 2.7
            , 4.0]
        self._constraints = [False, False] + [True] * 12
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        raise NotImplementedError

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "reduced_tb"
            elif environment == 'bopfox-ham':
                name = "reduced_tb"
            elif environment == 'bopfox-rep':
                name = "pp_reduced_tb"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


###########################################################################
class screenedpowerlaw_polynom5mod(Function):
    """    
    maximum number of parameters = 10

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [-0.3, 2.8, 2.4, 5.0, 5.6, 10.0, 10.1]
        self._numbers = [-0.3, 2.8, 2.4, 5.0, 5.6, 10.0, 10.1, 0.0, 0.0, 0.0]
        self._constraints = [True] * 7 + [False] * 3
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        raise NotImplementedError

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "screenedpowerlaw_polynom5mod"
            elif environment == 'bopfox-ham':
                name = "screenedpowerlaw_polynom5mod"
            elif environment == 'bopfox-rep':
                name = "pp_screenedpowerlaw_polynom5mod"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def print_function(self):
        raise NotImplementedError


class sum_funcs(Function):
    def __init__(self, **kwargs):
        self.functions = []
        self._parameters = []
        self._constraints = []
        self._numbers = []
        self._name = None
        self.set(**kwargs)

    def __call__(self, x):
        if isinstance(x, float):
            y = 0
        else:
            y = np.zeros(len(x))
        for f in self.functions:
            y += f(x)
        return y

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key.lower() == 'functions':
                self.functions = val
            elif key.lower() == 'parameters':
                self._parameters = val
            elif key.lower() == 'constraints':
                self._constraints = val
            elif key.lower() == 'numbers':
                self._numbers = val
            elif key.lower() == 'name':
                self._name = val
        if len(self.functions) != 0:
            par, con, num = [], [], []
            nn = 0
            np = 0
            for i in range(len(self.functions)):
                NN = len(self.functions[i].get_numbers())
                if len(self._constraints) != 0:
                    self.functions[i].constraints = self._constraints[nn:nn + NN]
                else:
                    con += self.functions[i].get_constraints()
                if len(self._parameters) != 0:
                    Np = self.functions[i].get_constraints().count(True)
                    self.functions[i].parameters = self._parameters[np:Np]
                    np += Np
                else:
                    par += self.functions[i].get_parameters()
                if len(self._numbers) != 0:
                    self.functions[i].numbers = self._numbers[nn:nn + NN]
                else:
                    num += self.functions[i].get_numbers()
                nn += NN
            self._parameters = par
            self._constraints = con
            self._numbers = num
        if len(self._constraints) != len(self._numbers):
            msg = 'Length of constraints should be %d' % len(self._numbers)
            raise ValueError(msg)
        if len(self._parameters) != self._constraints.count(True):
            msg = 'Mismatch between number of parameters and constraints.'
            raise ValueError(msg)

    def set_parameters(self, p):
        self._parameters = list(p)
        count = 0
        numbers = []
        for func in self.functions:
            par = []
            for i in range(len(func._constraints)):
                if func._constraints[i]:
                    par.append(self._parameters[count])
                    count += 1
            func.set_parameters(par)
            numbers += func.get_numbers()
        self._numbers = list(numbers)

    def get_name(self, environment="bopfox-ham"):
        if self._name is None:
            if environment == 'generic':
                name = "sum_funcs"
            elif environment == 'bopfox-ham':
                name = "sum_exp"
            elif environment == 'bopfox-rep':
                name = "sum_exp"
            else:
                raise NotImplementedError
            self._name = name
        return self._name

    def get_functions(self):
        return self.functions

    def print_function(self):
        s = ''
        for func in self.functions:
            s += ('+' + func.print_function())
        return s


class constant(Function):
    """
    constant

    maximum number of parameters = 1

    Note: Future functions should at least have the
          same attributes.
    """

    def __init__(self, **kwargs):
        self._parameters = [1]
        self._numbers = [1]
        self._constraints = [True]
        self._name = None
        self.set(**kwargs)

    def __call__(self, x, numbers=None):
        if numbers is None:
            numbers = self.get_numbers()
        return numbers[0]

    def get_name(self, environment="bopfox"):
        if self._name is None:
            if environment == 'generic':
                name = "constant"
            elif environment == 'bopfox':
                name = "pp_constant"
            self._name = name
        return self._name

    def print_function(self):
        return '%s' % self.get_numbers()[0]


class bounds:
    """
    Defines bound for the variables
    """

    def __init__(self, size, lo=-10, hi=10.0, **kwargs):
        xlo = np.ones(size) * lo
        xhi = np.ones(size) * hi
        self.xmin = np.array(xlo)
        self.xmax = np.array(xhi)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def print_fun(x, f, accepted):
    print(("at minimum %.4f accepted %d" % (f, int(accepted))))


def fit_sum(x, y, func, tol='default', fit_all=False, debug=False
            , minimizer='leastsq', constrain=False):
    def small_i(x, small):
        for i in range(len(x)):
            if x[i] >= small:
                return i

    param = []
    funcs = func.get_functions()
    # fit first exponential
    if tol == 'default':
        tol = max(abs(y)) / 300.
    param_exp, small = fit(x, y, funcs[0], tol=tol, fulloutput=True
                           , minimizer=minimizer, constrain=constrain)
    param += list(param_exp)
    mod = residual_beta(np.ones(len(param_exp)), y, x, funcs[0], param_exp)
    funcs[0].set_parameters(param_exp)
    if debug:
        import matplotlib.pyplot as pl
        pl.plot(x, y, 'ro')
        pl.plot(x, funcs[0](x), 'b-')
        pl.show()
        raise
    # loop over additional terms -p0*exp(-p1*x^p2)
    for i in range(1, len(funcs)):
        if not fit_all:
            si = small_i(x, small)
        else:
            si = len(x)
        if si < 3:
            param_exp2 = np.array(funcs[i].get_parameters()) * 0.0
        else:
            # generate a nice guess
            # tempf = exponential()
            # p0 = list(fit(x[:si],mod[:si],tempf,tol=0.05))
            # p0 +=[1]
            # funcs[i].set_parameters(p0)
            # fit it to modulus
            param_exp2, small = fit(x[:si], mod[:si], funcs[i], tol=tol
                                    , fulloutput=True, minimizer=minimizer
                                    , constrain=constrain)
            if i == 1 and debug:
                print((tol, param_exp2))
                pl.plot(x, mod, 'ro')
                pl.plot(x, funcs[i](x), 'r-')
                pl.show()
                raise
            # update modulus       
            mod = residual_beta(np.ones(len(param_exp2)), mod, x, funcs[i]
                                , param_exp2)
        param += list(param_exp2)
    func.set_parameters(param)
    if debug:
        pl.plot(x, y, 'ro')
        pl.plot(x, func(x), 'b-')
        pl.show()
        raise
    return param


###########################################################################
def fit(x, y, func, tol=1.0, fulloutput=False, debug=False, minimizer='leastsq'
        , constrain=False, fit_all=False):
    """
    Fit x-dependence of input function (func) to y. Since 
    the quality of the fit might be poor at short 
    distances, the minimum distance is adjusted until the 
    root-mean-square error of the fit is smaller than the 
    tolerance (tol).

    The fitting is carried out using the least-squares routine
    from scipy.
    """
    # print_format('Fitting %s'%func.get_name(),level=2)
    small = min(x)
    step = abs(x[1] - x[0])
    rms = 10
    p0 = func.get_parameters()
    if tol is None:
        fit_all = True
        tol = 9
    while rms > tol:
        newx = []
        newy = []
        for i in range(len(x)):
            if not fit_all and x[i] <= small:
                continue
            newx.append(x[i])
            newy.append(y[i])
        newx = np.array(newx)
        newy = np.array(newy)
        if minimizer == 'leastsq':
            param, cov_x, infodict, mesg, ier = \
                optimize.leastsq(residual_beta, np.ones(len(p0)), maxfev=3000
                                 , full_output=1, args=(newy, newx, func, p0, True
                                                        , constrain), epsfcn=0.000001)
            rms = np.sqrt(np.average(infodict['fvec'] ** 2))
        elif minimizer == 'basinhopping':
            res = optimize.basinhopping(residual_beta, np.ones(len(p0))
                                        , minimizer_kwargs={'args': (newy, newx, func, p0, False
                                                                     , constrain)}
                                        , T=10., stepsize=0.01
                                        # ,accept_test=bounds(len(p0))
                                        # ,callback=print_fun
                                        # ,niter=1000
                                        )
            rms = res.fun
            param = res.x
        elif 'nlopt' in minimizer.lower():
            import nlopt
            if 'neldermead' in minimizer.lower():
                nloptmin = nlopt.LN_NELDERMEAD
            elif 'bobyqa' in minimizer.lower():
                nloptmin = nlopt.LN_NELDERMEAD
            else:
                raise ValueError('No options for %s' % minimizer)
            opt = nlopt.opt(nloptmin, len(p0))
            residual = cresidual_beta(y=newy, x=newx, func=func, p0=p0
                                      , array=False, constrain=constrain)
            opt.set_min_objective(residual)
            param = opt.optimize([1] * len(p0))
        else:
            raise ValueError('No options for %s' % minimizer)
        param = np.array(param) * np.array(p0)
        param = list(param)
        small = small + step
        if fit_all:
            break
        if len(newx) <= len(p0):
            print_format("Desired tolerance not reached", level=3)
            print_format("Current rms is %8.4f" % rms, level=3)
            break
    if debug:
        import matplotlib.pyplot as pl
        pl.plot(x, y, 'ro')
        pl.plot(x, func(x), 'b-')
        pl.show()
        raise
    if fulloutput:
        return param, small
    else:
        return param


class cresidual_beta:
    def __init__(self, **kwargs):
        self.array = True
        self.constrain = False
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, val in list(kwargs.items()):
            if key == 'y':
                self.y = val
            elif key == 'x':
                self.x = val
            elif key == 'func':
                self.func = val
            elif key == 'p0':
                self.p0 = val
            elif key == 'array':
                self.array = val
            elif key == 'constrain':
                self.constrain = val

    def __call__(self, x, grad):
        p = np.array(x) * np.array(self.p0)
        self.func.set_parameters(p)
        err = self.y - self.func(self.x)
        if self.array:
            return err
        else:
            return np.sqrt(np.average(err ** 2))


def residual_beta(w, b, x, func, p0, array=True, constrain=False):
    """
    Objective function for the bond integral fitting.
    """
    if constrain:
        # this should be removed as bounds should be handled by optimizer
        bound = bounds(len(p0))
        if not bound(x_new=w):
            if array:
                return np.ones(len(x)) * 1E99
            else:
                return 1E99
    p = np.array(w) * np.array(p0)
    func.set_parameters(p)
    err = b - func(x)
    # print w,p,np.average(err**2)
    if array:
        return err
    else:
        return np.sqrt(np.average(err ** 2))


def _all_funcs():
    all_funcs = list(globals().items())
    funcs = []
    for key, val in all_funcs:
        try:
            if issubclass(val, Function) and key != 'Function':
                funcs.append(val)
        except:
            continue
    return funcs


def list_to_func(par, name, constraints=None):
    """
    Converts bopfox list of parameters to function.
    """
    funcs = _all_funcs()
    found = False
    name = name.lower()
    for func in funcs:
        for env in ['bopfox-ham', 'bopfox-rep']:
            f = func()
            try:
                fname = f.get_name(environment=env)
            except:
                continue
            if fname == name and len(f.get_numbers()) == len(par):
                found = True
                break
        if found:
            break
    if constraints is None:
        constraints = [True] * len(par)
        num = list(par)
    else:
        num = list(par)
        par = [num[i] for i in range(len(constraints)) if constraints[i]]
    if not found:
        func = None
    else:
        func = func(parameters=par, constraints=constraints, numbers=num
                    , name=name)
    return func


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    x = np.linspace(2, 5, 10)
    a = 103.54768901
    b = 1.834906789
    n = 1
    f = exponential(numbers=[a, b], parameters=[a, b], constraints=[True, True])
    g = exponential(parameters=[50, 1], numbers=[50, 1]
                    , constraints=[True, True])
    # f = GSP(numbers=[-1.9167,2.7411,1.0000,4.0000,4.0000,0.0500,0.0000])
    # g = GSP(parameters=[-2,2],numbers=[1,2.7411,1.0000,4.0000,4.0000
    #                                  ,0.0500,0.0000]
    #       ,constraints=[True,True,False,False,False,False,False])
    param = fit(x, f(x), g, minimizer='NLOPT.LN_NELDERMEAD')
    g.set_parameters(param)
    pl.plot(x, f(x), 'o', label='data')
    pl.plot(x, g(x), '-', label='fit')
    pl.xlabel('x')
    pl.ylabel('y')
    pl.legend(loc='best')
    pl.show()
    print(('a = %3.5f == %3.5f' % (param[0], a)))
    print(('b = %3.5f == %3.5f' % (param[1], b)))
