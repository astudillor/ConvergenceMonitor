#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np

__all__ = ['Convergence', 'IDENTITY', 'EUCLIDEAN_NORM', 'WEIGHTED_NORM']


def IDENTITY(x): return x


def EUCLIDEAN_NORM(x): return np.linalg.norm(x)


def WEIGHTED_NORM(G): return lambda x: np.dot(G.matvec(x), x)


class Convergence:
    """
    Convergence object motorizes the residual convergence though the callback
    function
    """

    def __init__(self, only_counter_iters=False, action=IDENTITY,
                 Norm=EUCLIDEAN_NORM, verbose=False, increment=1,
                 label='no_set'):
        self.iter_ = 0
        self.resVec = np.array([])
        self.action = action
        self.Norm = Norm
        self.verbose = verbose
        self.increment = increment
        self.only_counter_iters = only_counter_iters
        self.label = label

    def __call__(self, x):
        self.callback(x)

    def callback(self, x):
        self.iter_ += self.increment

        if self.only_counter_iters:
            return

        if self.action is not None:
            rnrm = self.action(x)
            if self.Norm is not None:
                rnrm = self.Norm(rnrm)
            self.resVec = np.append(self.resVec, rnrm)
            if self.verbose:
                print("{0}\t{1}".format(self.iter_, rnrm))

    def toFile(self, filename, header="x\ty\n"):
        try:
            with open(filename, 'w') as fhandle:
                fhandle.write(header)
                i = 0
                for res in self.resVec:
                    fhandle.write("{0}\t{1}\n".format(i, res))
                    i += self.increment
        except IOError:
            print("Unable to open file {0}".format(filename))

    def reset(self, only_counter_iters=False, action=IDENTITY,
              Norm=EUCLIDEAN_NORM, verbose=False, increment=1,
              label='no_set'):
        self.iter_ = 0
        self.resVec = np.array([])
        self.action = action
        self.Norm = Norm
        self.verbose = verbose
        self.increment = increment
        self.only_counter_iters = only_counter_iters
        self.label = label

    def finalResidualNorm(self):
        if len(self.resVec) == 0:
            return -1
        return self.resVec[-1]

    def toArray(self):
        return np.array(self.resVec)

    def printInfo(self):
        print(str(self))

    def scale(self, alpha):
        self.resVec = alpha*self.resVec

    def __str__(self):
        return "# of iters: {0}, residual {1}".format(self.iter_,
                                                      self.finalResidualNorm())

    def __len__(self):
        return len(self.resVec)

    def __getitem__(self, index):
        if self.index > len(self.resVec):
            return None
        return self.resVec[index]
