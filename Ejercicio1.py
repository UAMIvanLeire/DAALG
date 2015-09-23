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
 if cadena [len(cadena)-1::-1] == cadena:
  return True
 return False


#Pruebas funciones básicas
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



