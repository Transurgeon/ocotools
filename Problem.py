import numpy as np
import cvxpy as cp
from Barrier import *

class Problem:

    def __init__(self, f=None, grad=None):
        self.f = f
        self.grad = grad

    def increment(self):
        pass
    def optimal(self):
        pass
    def setF(self, f):
        self.f = f
    def setGrad(self, grad):
        self.grad = grad


class LinConsProblem(Problem):

    def __init__(self, n=1):
        super().__init__(self)
        self.n = n
        self.A = np.zeros((0, n))
        self.b = np.zeros((0, 1))
        self.baseB = self.b

    def getA(self):
        return self.A
    
    def getb(self):
        return self.b

    def setA(self, A):
        self.A = A

    def setb(self, b):
        self.b = b
        self.baseB = b

    def violation(self, x):
        return np.linalg.norm(self.A@x-self.b.T)

class LinearProgram(LinConsProblem):

    def __init__(self, n=None, A=None, b=None, C=None, d=None, c=None):
        super().__init__(n)
        if A is not None: self.setA(A) 
        if b is not None: self.setb(b) 
        self.C = np.zeros((0, n)) if C is None else C
        self.d = np.zeros((0, 1)) if d is None else d
        self.c = np.zeros((1, n)) if c is None else c
        self.grad = self.gradfunc
        self.f = self.func
    

    def getC(self):
        return self.C
    def getd(self):
        return self.d
    def getc(self):
        return self.C
    def setC(self, C):
        self.C = C
    def setc(self, c):
        self.c = c
    def setd(self, d):
        self.d = d

    def violation(self, x):
        return super().violation(x)+np.linalg.norm(np.max(self.C@x-self.d.T, 0))
    
    def loss(self, x):
        return self.c@x
    
    def gradfunc(self, x):
        return self.c.T
    
    def func(self, x):
        return self.c.dot(x.T)
    
    def getH(self, x):
        return np.zeros((self.n, self.n))
    
    def optimal(self):
        prob, x = self.optProblem()
        prob.solve()
        return prob.value, x.value
    
    def optProblem(self):
        x = cp.Variable((self.n, 1))
        obj = cp.Minimize(self.c@x)
        constraints = [self.A@x==self.b.T, self.C@x <= self.d.T]
        prob = cp.Problem(obj, constraints)
        return prob, x

class OCOMPC(LinearProgram):

    def __init__(self, n=1, A=None, b=None, C=None, d=None, c=None):
        super().__init__(n, A, b, C, d, c)
        self.setBarrier()

    def setBarrier(self):
        self.barr = BarrSum(self.n)
        for col in range(self.C.shape[0]):
            self.barr += LinLogBar(self.C[[col], :], self.d[:,[col]])

    def barrGradHess(self, x):
        return self.barr.grad(x), self.barr.hess(x)

    
    
        

           

def sigmoid(x):
    return 1/(1+np.e**-(x))

def sigmoidGrad(x):
    return sigmoid(x)**2 *np.e**-(x)

def sigmoidHess(x):
    return np.e**(-x)*(2*np.e**(-x)*sigmoid(x)**3)-sigmoidGrad(x)



class NetFlow(LinConsProblem):

    def __init__(self, n=1):
        super().__init__(n)
        self.f = self.loss
        self.grad = self.gradFct
        self.scaling = np.ones((n, 1))
        self.alpha = np.ones((n, 1))
        self.beta = np.zeros((n, 1))

    def setA(self, conjMatrix):
        pass

    def loss(self, x):
        return np.sum(self.scaling*
            (sigmoid(self.alpha*x+self.beta)+sigmoid(-self.alpha*x)))
    
    def gradFct(self, x):
        return self.alpha*(sigmoidGrad(self.alpha*x+self.beta)-sigmoidGrad(-self.alpha*x))

    def hessFct(self, x):
        return self.alpha**2*(sigmoidHess((self.alpha*x+self.beta)*np.identity(self.n))+
                sigmoidHess(-self.alpha*x*np.identity(self.n)))
    
    def getH(self, x):
        return self.hessFct(x)

class ConvNetFlow(LinConsProblem):

    def __init__(self, n=1):
        super().__init__(n)
        self.f = self.loss
        self.grad = self.gradFct
        self.c = np.ones((n, 1))
        self.alpha = np.ones((n, 1))
        self.beta = np.zeros((n, 1))
        self.T = 1

    def setLossParams(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def loss(self, x):
        return np.sum(self.alpha*x**2+self.beta*x+self.c)
    
    def gradFct(self, x):
        return self.alpha*2*x+self.beta

    def hessFct(self, x):
        return np.identity(self.n) *2* self.alpha
    
    def getH(self, x):
        return self.hessFct(x)

    def optimal(self):
        x = cp.Variable((self.n, 1))
        cost = cp.sum(cp.multiply(self.alpha, x**2)+cp.multiply(self.beta,x)+self.c)
        constraints = []
        for row in range(self.A.shape[0]):
            constraints.append(cp.sum(cp.multiply(self.A[row].reshape(self.n,1), x)) == self.b[row])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver="GUROBI")
        return prob.value, x.value

    def optProblem(self):
        x = cp.Variable((self.n, 1))
        cost = cp.sum(cp.multiply(self.alpha, x**2)+cp.multiply(self.beta,x)+self.c)
        constraints = []
        for row in range(self.A.shape[0]):
            constraints.append(cp.sum(cp.multiply(self.A[row].reshape(self.n,1), x)) == self.b[row])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        return prob

    def randomIncrement(self):
        self.alpha = np.random.sample(self.alpha.shape)*5

    def subIncrement(self):
        self.T += 1
        self.alpha = np.ones(self.alpha.shape) + np.random.sample(self.alpha.shape)*10/self.T

    def increment(self):
        self.randomIncrement()


class ExpNetFlow(LinConsProblem):

    def __init__(self, n=1):
        super().__init__(n)
        self.f = self.loss
        self.grad = self.gradFct
        self.c = np.ones((n, 1))
        self.alpha = np.ones((n, 1))
        self.beta = np.ones((n, 1))/10
        self.T = 1

    def setLossParams(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def loss(self, x):
        return np.sum(self.alpha*np.e**(self.beta*x)+self.c)
    
    def gradFct(self, x):
        return self.alpha*self.beta*np.e**(self.beta*x)

    def hessFct(self, x):
        return np.identity(self.n) * self.alpha*self.beta**2*np.e**(self.beta*x)
    
    def getH(self, x):
        return self.hessFct(x)

    def optimal(self):
        x = cp.Variable((self.n, 1))
        cost = cp.sum(cp.multiply(self.alpha, cp.exp(cp.multiply(x, self.beta)))+self.c)
        constraints = []
        for row in range(self.A.shape[0]):
            constraints.append(cp.sum(cp.multiply(self.A[row].reshape(self.n,1), x)) == self.b[row])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        return prob.value, x.value

    def randomIncrement(self, scaling=100):
        self.alpha = np.random.sample(self.alpha.shape)*scaling
        self.beta = np.random.sample(self.beta.shape)/10+1/10

    def subIncrement(self, scaling=100):
        self.alpha = np.random.sample(self.alpha.shape)*scaling/np.sqrt((self.T+20)**1.75)+1
        self.beta = np.random.sample(self.beta.shape)/10/np.sqrt((self.T+20)**1.75)+1/10
        self.b = self.baseB+np.random.sample(self.baseB.shape)*scaling/np.sqrt((self.T+20)**1.75)
        self.T += 1

    def increment(self):
        self.subIncrement()



class QLCP(Problem):

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.increment()

    def increment(self):
        Q = np.random.random((self.n, self.n))
        Q = (Q + Q.T)/2
        def f(x):
            return float(x.T@Q@x)
        def grad(x):
            return 2*Q@x
        self.f = f
        self.grad = grad
        self.H = Q
        m = np.random.randint(1,5)
        self.A = np.random.random((m, self.n))
        self.b = np.random.random((m, 1))
    
    def getA(self):
        return self.A
    def getb(self):
        return self.b
    def getH(self, x):
        return self.H