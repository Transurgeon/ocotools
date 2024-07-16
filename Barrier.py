import numpy as np
import scipy.sparse as sp

class Barrier:

    def __init__(self, n):
        self.n = n

    def __add__(self, other):
        if other.n == self.n:
            ans = BarrSum(self.n)
            ans.fls.append(self)
            if isinstance(other, BarrSum):
                for func in other.fls:
                    ans.fls.append(func)
            else:
                ans.fls.append(other)
            return ans
        return None

class LogBar(Barrier):

    def __init__(self, b, loc, le=True):
        super().__init__(len(b))
        self.b = b
        self.loc = loc
        self.sign = -1 if le else 1
        self.P = np.zeros((self.n, 1))
        for i in loc:
            self.P[i] = 1

    def setb(self, b):
        self.b = b

    def logf(self, x):
        ans = 0
        for i in self.loc:
            ans += -np.log(self.sign*(x[i]-self.b[i]))
        return ans
    
    def f(self, x):
        ans = 0
        for i in self.loc:
            ans += -1*self.sign*(x[i]-self.b[i])
        return ans
    
    def grad(self, x):
        try:
            return -1/(x-self.b)*self.P
        except:
            return np.zeros(x.shape)
    
    def hess(self, x):
        return np.identity(self.n)*1/(x-self.b)**2*self.P
    
    def isFeasible(self, x):
        for i in self.loc:
            if -1*self.sign*(x[i]-self.b[i]) > 0 :
                return False
        return True

class LinLogBar(Barrier):

    def __init__(self, a, b):
        super().__init__(max(a.shape))
        self.a = a
        self.b = b
    
    def f(self, x):
        return self.a.dot(x.T)-self.b
    def logf(self, x):
        return -np.log(self.f(x))
    def grad(self, x):
        try:
            return -1/(self.a.dot(x.T)-self.b)*self.a.T
        except:
            return sp.csr_array(x.shape)
        
    def hess(self, x):
        return 1/(self.a.dot(x.T)-self.b)**2*self.a.T@self.a
    def isFeasible(self, x):
        return self.f(x) <=0



class QuadBar(Barrier):

    def __init__(self, n, A, B):
        super().__init__(n)
        self.A = A
        self.B = B

    def f(self, x):
        return (x.T@self.A@x+self.B.T@x).flatten()

    def grad(self, x):
        val = x.T@self.A@x+self.B.T@x
        return -(2*self.A@x + self.B)/val
    
    def hess(self, x):
        val = x.T@self.A@x+self.B.T@x
        grad = 2*self.A@x + self.B
        return -1/val*2*self.A + grad@grad.T/val**2
    
    def isFeasible(self, x):
        return self.f(x) < 0
        

class BarrSum(Barrier):

    def __init__(self, n):
        super().__init__(n)
        self.fls = []

    def f(self, x):
        ans = 0
        for func in self.fls:
            ans += func.f(x)
        return ans

    
    def grad(self, x):
        ans = np.zeros((self.n, 1))
        for func in self.fls:
            ans += func.grad(x)
        return ans
    
    def hess(self, x):
        ans = sp.csr_array((self.n, self.n))
        for func in self.fls:
            ans += func.hess(x)
        return ans

    def __iadd__(self, other):
        if isinstance(other, Barrier):
            if other.n == self.n:
                if isinstance(other, BarrSum):
                    for func in other.fls:
                        self.fls.append(func)
                else:
                    self.fls.append(other)
            return self
        else:
            raise TypeError('Cannot add a non-Barrier function')

    def __add__(self, other):
        if other.n == self.n:
            ans = BarrSum(self.n)
            for func in self.fls:
                ans.fls.append(func)
            if isinstance(other, BarrSum):
                for func in other.fls:
                    ans.fls.append(func)
            else:
                ans.fls.append(other)
            return ans
        return None

    def isFeasible(self, x):
        for func in self.fls:
            if not func.isFeasible(x):
                return False
        return True