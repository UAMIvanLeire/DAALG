import numpy as np
import random
import matplotlib.pyplot as mpl
import timeit as timeit
import math
import fit
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys





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

def isPalyndrome(cadena):
 S = initS()
 val = len(cadena)
 for i in range(val):
  push(cadena[i], S)
 i = 0
 while(i <= val-1):
  a = pop(S)
  if a != cadena[i]:
   return False
  i = i +1
 return True

def isPalyndromeSmall(cadena):
 return cadena [len(cadena)-1::-1] == cadena

def randomString(long, strBase):
  list = initS();
  for i in range(long):
    push(random.choice(strBase), list)
  return list

def randomPalyndrome(long, strBase):
  if (long%2) == 0:
    list1 = randomString(long / 2, strBase)
    list2 = list1 [len(list1)-1::-1]
    return list1 + list2
  else:
    list1 = randomString(long / 2, strBase)
    list2 = list1 [len(list1)-1::-1]
    list3 = randomString(1, strBase)
    return list1 + list3 + list2

#FIX-ME
def generateRandomPalyndrList(numSt, longMax, probPalyndrome):
  return 0

def countPalyndromesList(l):
  laux = initS()
  list = initS()
  for i in range(len(l)):
    eleml = pop(l)
    push(eleml, laux)
    if isPalyndromeSmall(eleml):
      push(eleml, list)
  print laux
  l = laux
  return len(list)

def list2file(l, fName):
  f = open(fName, "w")
  l = l[len(l)-1::-1]
  for i in range(len(l)):
    f.write(pop(l))
    f.write("\n")
  f.close()
  return

def countLines(fName):
  return len(open(fName).readlines())

def countPalyndromesFile(fName):
  list = initS()
  f = open(fName)
  for line in f.read().splitlines():
    push(line, list)
  return countPalyndromesList(list)

def permutacion(sizeP):
  return np.random.permutation(sizeP)

def checkPerms(numPerms, sizeP):
  matrix = np.zeros(shape = (numPerms, sizeP))
  for i in range(numPerms):
    matrix[i] = permutacion(sizeP)

  mpl.hist(matrix)
  mpl.show()
  return matrix


fName = sys.argv[1]
numPalyndr = countPalyndromesFile(fName)
numLines = countLines(fName)

print("fName %s:\tnum sequences: %d\tnum palyndromes: %d" % (fName, numLines, numPalyndr)
  