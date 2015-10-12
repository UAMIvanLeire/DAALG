import numpy as np
import random
import matplotlib.pyplot as mpl
import timeit as timeit
import math
import fit
import matplotlib.pyplot as plt
from sklearn import linear_model



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

def qs(t, p, u):
  
  lsIgual = initS()
  lsMayor = initS()
  lsMenor = initS()
  if u > 1:
    pivot = t[0]
    for i in t:
      if i > pivot:
        lsMayor.append(i)
      if i < pivot:
        lsMenor.append(i)
      if i == pivot:
        lsIgual.append(i)
    print t
    #print "Pivote %s Mayor %s Igual %s Menor %s  " % (pivot, lsMayor, lsIgual, lsMenor )
    return qs(lsMayor,0,len(lsMayor)-1) + lsIgual + qs(lsMenor,0,len(lsMenor)-1)
  else:
    return t


#FIX-ME
def qs2(t, p, u):
  
  return t;

#FIX-ME
#Preguntar profesor como hacer siguiente ejercicio si no tengo la funcion partir


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

#return timeList

#Primera prueba que funciona
  ##xList = zip(timeList, valueList)

  ##regr = linear_model.LinearRegression()
  ##linearModel = regr.fit(xList, valueList)
  
#Prueba2

  # print "hello"
  # xList = zip(timeList, valueList)
  # print xList
  # xList = np.array(xList)
  # print xList

  # print np.shape(xList)
  # print xList.dtype
  # nptime = np.array(timeListAux).reshape(len(timeList), 1)


  # regr = linear_model.LinearRegression()
  # linearModel = regr.fit(nptime, valueList)

  #Prueba codigo nuevo

  S = timeList
  L = valueList
  N = len(valueList) 

  lr = linear_model.LinearRegression()
  SS = np.array(S).reshape(N, 1)
  print "SS"
  print SS
  print "FIT"
  NM = lr.fit(SS, np.array(L))
  print NM
  LP = lr.predict(SS)
  print "PREDICT"
  print LP

  print "PLOT"
  
  
  plt.plot(S, LP, '-', S, L, '.')
  plt.show()
  # plt.plot(timeList)
  # plt.ylabel("Normal values")
  # plt.show()

  # plt.plot(LP)
  # plt.ylabel("Prediction")
  # plt.show()

  #Prediction

  #print regr.predict(xList)
  return 

# def func2fit(nPerms, sizeIni, sizeFin, step):
#   #Cambiar cuando este qs2
#   return np.array([timeSort(qs, nPerms, sizeIni, sizeFin, step), timeSort(qs, nPerms, sizeIni, sizeFin, step)])
  


# def fitPlot(l, func2fit, nNodesIni, nNodesFin, step):
#   resultTimeValues = func2fit()

#   timeList = func2fit()

#   return 

#Pruebas funciones basicas
list = initS()
print emptyS(list)
push(1, list)
push(2, list)
push(3, list)
print list
print emptyS(list)
print list
print pop(list)
print list
print pop(list)
print list

#Pruebas isPalyndrome
print "Palyndrome True"
print isPalyndrome("hooh")
print "Palyndrome False"
print isPalyndrome("fdgfdf")
print "Palyndrome True"
print isPalyndrome("555666555")
print "Palyndrome False"
print isPalyndrome("hola")

#Pruebas isPalyndromeSmall
print isPalyndromeSmall("hello")
print isPalyndromeSmall("helloolleh")

#Pruebas randomString
print "Prueba randomString con cadena \"pali\""
lista = ['p', 'a', 'l', 'i']
print randomString(4, lista)
print len(randomString(4,lista))
print randomString(10, lista)

#Pruebas randomPalyndrome
print "Prueba randomPalyndrome con cadena \"pali\" y long impar"
print randomPalyndrome(5, lista)
print "Prueba randomPalyndrome con cadena \"pali\" y long par"
print randomPalyndrome(6, lista)

#Pruebas countPalyndromesList
print "Prueba countPalyndromesList"
list = ["hola", "hooh", "lola", "ollo", "miim"]
print  countPalyndromesList(list)


#Pruebas list2file
print "Prueba list2file en archivo practica 1"
list = ["hola", "hooh", "lola", "ollo", "miim"]
list2file(list, "practica.txt")

#Prueba countPalyndromeFile:
print "Prueba countPalyndromesFile"
print countPalyndromesFile("practica.txt")

#Prueba permutacion
print "Prueba permutacion"
print permutacion(20)

#Prueba checkPerms
print "Prueba checkPerms"
print checkPerms(3000, 10)

#Prueba qs 
print "Prueba qs"
perm = permutacion(10)
print qs(perm, 0, 9)

#Prueba timeSort(sortM, nPerms, sizeIni, sizeFin, step)
print "Prueba timeSort"
#print timeSort("qs", 10, 1, 100, 10)



def fitPlot(l, func2fit, nNodesIni, nNodesFin, step):
  func2fit("qs", 10, 1, 100, 10)
  func2fit("qs", 10, 1, 100, 10)
  def hola():
    print l
    return

  hola()
  return
fitPlot(1, timeSort, 1, 100, 10)

