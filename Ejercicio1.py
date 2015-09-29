from numpy import *


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

def generateRandomPalyndrList(numSt, longMax, probPalyndrome):
  return 0

def countPalyndromesList(l):
  laux = l
  list = initS()
  for i in range(len(l)):
    eleml = pop(l)
    if isPalyndromeSmall(eleml):
      push(eleml, list)
  l = laux
  return len(list)

def list2file(l, fName):
  f = open(fName, "w")
  print len(l)
  for i in range(len(l)):
    f.write(pop(l))
    f.write("\n")
  f.close()
  return


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
list2file(list, "practica.txt")
