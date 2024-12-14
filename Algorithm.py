import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from ocotools.Problem import ConvNetFlow, ExpNetFlow

class Algorithm:
    """
    An OCO algorithm. Is instantiated with the number of dimensions and (optional) hyper-parameters.
    Must implement the $\text{update}$ function which does its OCO update.
    """
    def __init__(self, n):
        self.n = n
        self.x = sp.csr_array((n,1))
        self.name = ""

    def __str__(self):
        return self.name

    def update(self):
        pass
    
    def setX(self, x0):
        self.x = x0.reshape((self.n, 1))

    def getX(self):
        return list(self.x.reshape(self.n))

    def eval(self, prob):
        return prob.f(self.x)

    def __iadd__(self, other):
        if isinstance(other, float):
            other = int(other)
        if isinstance(other, int):
            for _ in range(other):
                self.update()

class MOSP(Algorithm):

    def __init__(self, n, alpha=0.3, omega=2):
        super().__init__(n)
        self.alpha = alpha
        self.om = omega
        self.dual = np.zeros((2*n, 1))
        self.name = "MOSP"

    def setDual(self, dual):
        self.dual = dual

    def violation(self, prob):
        A = prob.getA()
        b = prob.getb()
        return np.linalg.norm(A@self.x-b)

    def update(self, prob):
        A = prob.getA()
        b = prob.getb()
        bigA = np.vstack((np.vstack((A, -A)),
                            np.zeros((2*(self.n-A.shape[0]), A.shape[1]))))
        bigb = np.vstack((np.vstack((b, -b)),
                            np.zeros((2*(self.n-b.shape[0]), 1))))
        updateVec = self.om*(bigA@self.x-bigb)
        self.dual = np.maximum(self.dual+updateVec, 0)
        grad = prob.grad(self.x)
        self.x += -self.alpha*(grad+bigA.T@self.dual)
        return self.x

class MOSPBasic(Algorithm):
    def __init__(self, n, alpha=0.03, omega=1.7):
        super().__init__(n)
        self.alpha = alpha
        self.om = omega
        self.dual = np.zeros((n, 1))
        self.T = 1
        self.name = "MOSP"

    def setDual(self, dual):
        self.dual = dual

    def violation(self, prob):
        A = prob.getA()
        b = prob.getb()
        return np.linalg.norm(A@self.x-b)

    def update(self, prob):
        self.T += 1
        alpha = self.alpha*self.T**(-1/3)
        omega = self.om*self.T**(-1/3)
        A = prob.getA()
        b = prob.getb()
        bigA = np.vstack((A, np.zeros((self.n-A.shape[0], A.shape[1]))))
        bigb = np.vstack((b, np.zeros((self.n-b.shape[0], 1))))
        
        grad = prob.grad(self.x)
        self.x += -alpha*(grad+bigA.T@self.dual)

        updateVec = omega*(bigA@self.x-bigb)
        self.dual = np.maximum(self.dual+updateVec, 0)
        return self.x
    
class MOSPMod(Algorithm):
    def __init__(self, n, alpha=0.03, omega=1.7):
        super().__init__(n)
        self.alpha = alpha
        self.om = omega
        self.dual = np.zeros((n, 1))
        self.T = 1
        self.name = "MOSP"

    def setDual(self, dual):
        self.dual = dual

    def violation(self, prob):
        A = prob.getA()
        b = prob.getb()
        return np.linalg.norm(A@self.x-b)

    def update(self, prob):
        self.T += 1
        alpha = self.alpha*self.T**(-1/3)
        omega = self.om*self.T**(-1/3)
        A = prob.getA()
        b = prob.getb()
        bigA = np.vstack((A, np.zeros((self.n-A.shape[0], A.shape[1]))))
        bigb = np.vstack((b, np.zeros((self.n-b.shape[0], 1))))
        
        grad = prob.grad(self.x)
        self.x += -alpha*(grad+bigA.T@self.dual)

        updateVec = omega*(bigA@self.x-bigb)
        self.dual = np.maximum(self.dual+updateVec, 0)
        return self.x


class Lagrangian(Algorithm):

    def __init__(self, n, alpha=1, sigma=10):
        super().__init__(n)
        self.alpha = alpha
        self.sigma = sigma
        self.dual = np.zeros((n,1))
        self.name = "MALM"
        self.T = 1

    def update(self, prob):
        self.T += 1
        alpha = self.alpha*np.sqrt(self.T)
        sigma = self.sigma/np.sqrt(self.T) 
        A = prob.getA()
        b = prob.getb()
        bigA = np.vstack((A, np.zeros((self.n-A.shape[0], A.shape[1]))))
        bigb = np.vstack((b, np.zeros((self.n-b.shape[0], 1))))
        x = cp.Variable((self.n, 1))
        cost = 1/2/sigma*(cp.sum(cp.power(cp.pos(self.dual+sigma*(bigA@x-bigb)), 2))
                -np.sum(self.dual**2))+alpha/2*cp.sum(cp.power(x-self.x, 2))
        if isinstance(prob, ConvNetFlow):
            cost = cost + cp.sum(cp.multiply(prob.alpha, cp.power(x, 2))+cp.multiply(prob.beta, x)+prob.c)
        elif isinstance(prob, ExpNetFlow):
            cost = cost + cp.sum(cp.multiply(prob.alpha, cp.exp(cp.multiply(x, prob.beta)))+prob.c)
        convProb = cp.Problem(cp.Minimize(cost))
        convProb.solve()
        self.x = np.asarray(x.value).reshape((self.n, 1))
        self.dual = np.maximum(self.dual+self.sigma*(bigA@self.x-bigb), 0)


class OPENM(Algorithm):

    def __init__(self, n):
        super().__init__(n)
        self.name = "OPEN-M"

    def update(self, prob):
        A = prob.getA()
        b = prob.getb()
        self.x += A.T@np.linalg.inv(A@A.T)@(b-A@self.x)
        grad = prob.grad(self.x)
        H = prob.getH(self.x)
        D = np.vstack((np.hstack((H, np.transpose(A))), 
                np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))))
        augGrad = np.vstack((grad, np.zeros((A.shape[0],1))))
        delta = (np.linalg.inv(D)@augGrad)[:self.n]
        self.x -= delta
        return self.x

    def violation(self, prob):
        A = prob.getA()
        b = prob.getb()
        return np.linalg.norm(A@self.x-b)


class IPM(Algorithm):

    def __init__(self, n, nlam):
        super().__init__(n)
        self.nlam = nlam
        self.eta = 1
        self.name = "IPM"
        self.lamda = np.zeros((nlam, 1))
        self.damped = True

    def setEta(self, eta):
        self.eta = eta

    def setdamped(self, isdamped):
        self.damped = isdamped

    def update(self, prob):
        A = prob.getA()
        b = prob.getb()
        grad, H = prob.barrGradHess(self.x)
        D = sp.block_array([[H, A.T],[A, None]]) 
        grad = grad + self.eta*prob.grad(self.x)
        viol = A.dot(self.x.T)-b.T
        viol.data = np.round(viol.data, 8)
        augGrad = sp.vstack((grad, viol))
        delta = sp.linalg.spsolve(D, augGrad)
        norm = augGrad.T.dot(delta)
        if norm > 1 and self.damped:
            delta = delta/norm
        #print(norm)
        self.x -= delta[:self.n]
        self.lamda = delta[self.n:]
        return self.x
    
    def etaUpdate(self, prob, beta=1.02):
        self.update(prob)
        self.eta *= beta
        return self.update(prob)
    
    def setX(self, x):
        self.x = x
    
class IPMDamped(IPM):

    def update2(self, prob):
        A = prob.getA()
        b = prob.getb()
        grad, H = prob.gradHess(self.x)
        D = np.vstack((np.hstack((H, np.transpose(A))), 
                np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))))
        grad[-1] += self.eta
        augGrad = np.vstack((grad, np.round(A@self.x-b, 8)))
        delta = np.linalg.solve(D, augGrad)
        norm = augGrad.T@delta
        count = 0
        while(not prob.getSOCPBarr().isFeasible(self.x - delta[:self.n])):
            delta *= 0.8
            count += 1
        print(norm, count)
        self.x -= delta[:self.n]
        self.lamda = delta[self.n:]
        return self.x

    def etaUpdate(self, prob, beta=1.02):
        self.update2(prob)
        self.eta *= beta
        return self.update2(prob)
    
class IPMFeasible(IPM):

    def update2(self, prob):
        A = prob.getA()
        b = prob.getb()
        grad, H = prob.barrGradHess(self.x)
        D = sp.block_array([[H, A.T],[A, None]]) 
        grad = grad + self.eta*prob.grad(self.x)
        viol = A.dot(self.x.T)-b.T
        viol.data = np.round(viol.data, 8)
        augGrad = sp.vstack((grad, viol))
        delta = sp.linalg.spsolve(D, augGrad)
        norm = augGrad.T.dot(delta)
        count = 0
        while(not prob.barr.isFeasible(self.x - delta[:self.n])):
            delta *= 0.8
            count += 1
        print(norm, count)
        self.x -= delta[:self.n]
        self.lamda = delta[self.n:]
        return self.x

    def etaUpdate(self, prob, beta=1.02):
        self.update2(prob)
        self.eta *= beta
        return self.update2(prob)




class Function:

    def __init__(self, n=1):
        self.n = n

    def f(self, x):
        pass
    def grad(self, x):
        pass
    def hess(self, x):
        pass
    def cvxpyFunc(self):
        pass


class LinFunction(Function):

    def __init__(self, func, n=1, a=1, b=0):
        self.a = a
        self.b = 0
        self.func = func(n)

    def f(self, x):
        return self.func.f(self.a*x+self.b)

    def grad(self, x):
        return self.a*self.func.grad(self.a*x+self.b)

