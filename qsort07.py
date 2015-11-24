#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as mpl
import timeit as timeit
import math
import fit
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys
import argparse 


def initS():
 return []

def emptyS(S):
 if not S:
   return True
 else:
   return False

def push(elem, S):
 S.append(elem)

def pop(S):
 aux = S[len(S)-1]
 if emptyS(S):
  return False
 del S[len(S)-1]
 return aux


def permutacion(sizeP):
  return np.random.permutation(sizeP)

def checkPerms(numPerms, sizeP):
  matrix = np.zeros(shape = (numPerms, sizeP))
  for i in range(numPerms):
    matrix[i] = permutacion(sizeP)

  mpl.hist(matrix)
  mpl.show()
  return matrix

def partir(t, p, u, pivote):
  lsMayor = initS()
  lsMenor = initS()
  lsEqual = initS()
  if p!=u:
    for i in range(p, u+1):
      if t[i] > t[pivote]:
        lsMayor.append(t[i])
      if t[i] < t[pivote]:
        lsMenor.append(t[i])
      if t[i] == t[pivote]:
        lsEqual.append(t[i])
    return lsMenor , lsEqual , lsMayor
  return [],[t[p]],[]

def firstP():
  return 0

def qs(t, p, u):
  if len(t) == 0:
    return []
  less = initS()
  equal = initS()
  great = initS()
  less, equal, great = partir(t, 0, len(t) - 1, firstP())
  return qs(less,0,len(less)-1) + equal + qs(great,0,len(great)-1)



def qs2(t, p, u):
  if u-p <= 1:
    return t
  queue = initS()
  queue.append((p, u))
  while len(queue):
    rango = queue.pop()
    if rango[1] - rango[0] > 1:
      pivote = rango[0]
      for i in range(rango[0]+1,rango[1]+1):
        if t[i] < t[pivote]:
          t[i], t[pivote+1] = t[pivote+1], t[i]
          t[pivote+1], t[pivote] = t[pivote], t[pivote+1]
          pivote += 1
      queue.append((rango[0], pivote))
      queue.append((pivote+1, rango[1]))
  return t





def timeSort(sortM, nPerms, sizeIni, sizeFin, step):
  timeList = initS()
  valueList = initS()
  timeListAux = initS()
  
  aux = sizeIni
  while aux < sizeFin:
    setup = "from __main__ import %s as sortM, permutacion; permAux = permutacion(%d)" % (sortM, aux)
    time = timeit.timeit("sortM(permAux, 0, len(permAux))", setup=setup, number = nPerms)
    push(time, timeList)
    push(time, timeListAux)
    push(aux, valueList)
    aux = aux + step
    print "tamaño: %d t medio: %lf" % (aux, time)

  return timeList

def nlog(nNodesIni, nNodesFin, step):
  log = []
  for i in range(nNodesIni, nNodesFin, step):
    log.append(i * np.log10(i))
  nlog = np.array(log);
  return nlog

def fitPlot(l, func2fit, nNodesIni, nNodesFin, step):
  
  nlog = func2fit(nNodesIni, nNodesFin, step);
  print nlog
  N = len(l) 

  lr = linear_model.LinearRegression()
  lr.fit(nlog[:, np.newaxis], l)
  fit = lr.predict(nlog[:, np.newaxis])

  plt.plot(fit)
  plt.plot(l)
  plt.show()

  return


parser = argparse.ArgumentParser(description='Parseo de argumentos')
parser.add_argument('-nPerms', type=int)
parser.add_argument('-sizeI', type=int)
parser.add_argument('-sizeF', type=int)
parser.add_argument('-step', type=int)

args = parser.parse_args()

timeSort('qs', args.nPerms, args.sizeI, args.sizeF, args.step)


