import numpy as np
import json

import matplotlib.pyplot as plt


class Benchmark:

    def __init__(self, n, prob, *algo):
        self.n = n
        self.prob = prob(n)
        self._doInit(algo)

    def __init__(self, prob, *algo):
        self.n = prob.n
        self.prob = prob
        self._doInit(algo)
        

    def _doInit(self, algo):
        self.data = {}
        self.algo = []
        self.algoNames = []
        for alg in algo:
            algInst = alg(self.n)
            self.algoNames.append(str(algInst))
            self.algo.append(algInst)
            self.data[algInst.name] = {'regret':[0],'avgRegret':[0],
                                        'roundRegret':[0],'violation':[0]}
        self.T = 0
        self.Taxis = [0]
        self.optimal = [0]



    def increment(self, probIncrement=None, n=1):
        for _ in range(n):
            self.T += 1
            self.Taxis.append(self.T)
            self.optimal.append(self.prob.optimal()[0])
            for alg in self.algo:
                regret = np.abs(self.prob.optimal()[0]-alg.eval(self.prob))
                self.data[alg.name]['roundRegret'] += [regret]
                last = self.data[alg.name]['regret'][-1]
                self.data[alg.name]['regret'].append(last+regret)
                self.data[alg.name]['avgRegret'].append((last+regret)/self.T)
                last = self.data[alg.name]['violation'][-1]
                self.data[alg.name]['violation'].append(self.prob.violation(alg.getX())+last)
                alg.update(self.prob)

            if probIncrement is None:
                self.prob.increment()
            else:
                probIncrement()

    def __add__(self, n):
        if isinstance(n, int):
            for _ in range(n):
                self.increment()
        return self

    def __iadd__(self, n):
        if isinstance(n, int):
            for _ in range(n):
                self.increment()
        return self

    def dump(self, filename='bench.json'):
        with open(filename, 'w') as file:
            dct = {}
            dct['data'] = self.data
            dct['Taxis'] = self.Taxis
            dct['n'] = self.n
            dct['T'] = self.T
            dct['algorithms'] = self.algoNames
            for alg in self.algo:
                dct['data'][alg.name]['x'] = alg.getX()
            json.dump(dct, file)
            file.close()

    def load(self, filename='bench.json'):
        with open(filename,'r') as file:
            dct = json.load(file)
            self.data = dct['data']
            self.Taxis = dct['Taxis']
            self.T = dct['T']
            self.n = dct['n']
            self.algo = []
            self.algoNames = dct['algorithms']
            for alg in self.algoNames:
                klass = globals()[alg]
                algInst = klass(self.n)
                algInst.setX(np.asarray(self.data[alg]['x']))
                self.algo.append(algInst)
            file.close()

    def loadData(self, filename='bench.json'):
        with open(filename, 'r') as file:
            dct = json.load(file)
            self.data = dct['data']
            self.Taxis = dct['Taxis']
            self.T = dct['T']
            self.n = dct['n']
            self.algoNames = dct['algorithms']
            file.close()

    def plotRegret(self):
        for alg in self.algo:
            plt.plot(self.Taxis, self.data[alg.name]['regret'],linewidth=2.0,label=alg.name)

    def plotAvgRegret(self):
        for alg in self.algo:
            plt.plot(self.Taxis, self.data[alg.name]['avgRegret'],linewidth=2.0,label=alg.name)

    def plot(self, property):
        for alg in self.algo:
            if property in self.data[alg.name]:
                plt.plot(self.Taxis, self.data[alg.name][property],linewidth=2.0,label=alg.name)

    def plotOptimal(self):
        if len(self.Taxis)>0:
            plt.plot(self.Taxis[1:], self.optimal[1:])