#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import Queue as pq
import timeit as timeit
import math
import matplotlib.pyplot as plt
#from sklearn import linear_model
import argparse 

###############################################################################
# Nombre: genSuperCrec
# Funcionalidad: Genera una sucesión supercreciente con n_terms números.
# Argumentos: 
#             -n_terms: Número elementos de la sucesión a generar.
# Salida: 
#             -sucension: Sucesión supercreciente.
###############################################################################

def genSuperCrec(n_terms):
	sucesion = [];
	min = 0;
	for i in range(n_terms):
		sucesion.append(np.random.randint(min, min+100))
		min = np.sum(sucesion)
	return sucesion

###############################################################################
# Nombre: multiplier
# Funcionalidad: Encuentra un número primo relativo con mod y un número generado
# aleatoriamente y que sea superior a mult_ini.
# Argumentos: 
#             -mod: módulo.
#			  -mult_ini: número mínimo para evitar multiplicadores pequeños.
# Salida: 
#             -prim: Entero primo relativo a mod y mult_ini.
###############################################################################

def multiplier(mod, mult_ini):
	prim = np.random.randint(mult_ini, mod)
	while mcd(mod, prim) != 1:
		prim = np.random.randint(mult_ini, mod)
	return prim 



###############################################################################
# Nombre: mcd
# Funcionalidad: Implementación del algoritmo de Euclides para hallar el máximo
# común divisor entre x e y.
# Argumentos: 
#             -x: Elemento del par para el que hallar el máximo común divisor.
#			  -y: Elemento del par para el que hallar el máximo común divisor.
# Salida: 
#             -max: Máximo común divisor.
###############################################################################

def mcd(x, y):
	if x % y == 0:
		return y
	else:
		return mcd(y, x%y)

###############################################################################
# Nombre: inverse
# Funcionalidad: Calcula el inverso de p módulo mod
# Argumentos: 
#             -p: Elemento del par para el que hallar el inverso del módulo.
#			  -mod: Elemento del par para el que hallar el inverso del módulo.
# Salida: 
#             -inv: Inverso del módulo entre p y mod.
###############################################################################

def inverse(p, mod):
	b = 0
	x = 0
	while b < mod:
		x = (p * b) % mod
		if x == 1:
			return b
		b += 1

###############################################################################
# Nombre: modMultInv
# Funcionalidad: Calcular un módulo, inverso y multiplicador para la lista super
# creciente dada.
# Argumentos: 
#             -lSC: Sucensión supercreciente. 
# Salida: 
#             -mul: Multiplicador.
#			  -inv: Inverso.
# 			  -mod: Módulo.
###############################################################################

def modMultInv(lSC):
	mod = np.random.randint(np.sum(lSC))
	mul = multiplier(mod, 10)
	inv = inverse(mul, mod)
	return  mul, inv, mod

###############################################################################
# Nombre: genSucesionPublica
# Funcionalidad: Genera la solución pública asociada a la sucesión supercreciente
# , al multiplicador, y al módulo dados.
# Argumentos: 
#             -lSC: Sucensión supercreciente.
#			  -p: Multiplicador.
#			  -mod: Módulo. 
# Salida: 
#             -sol: Solución pública.
###############################################################################

def genSucesionPublica(lSC, p, mod):
	sol = []
	for k in lSC:
		sol.append(k*p%mod)
	return sol

###############################################################################
# Nombre: lPub_2_lSC
# Funcionalidad: Genera la solución privada asociada a la sucesión supercreciente
# , al multiplicador, y al módulo dados.
# Argumentos: 
#             -lSC: Sucensión supercreciente.
#			  -p: Multiplicador.
#			  -mod: Módulo. 
# Salida: 
#             -sol: Solución pública.
###############################################################################

def lPub_2_lSC(l_pub, q, mod):
	sol = []
	for k in l_pub:
		sol.append(k*q%mod)
	return sol

###############################################################################
# Nombre: genRandomBitString
# Funcionalidad: Genera un array de bits aleatorios.
# Argumentos: 
#             -n_bits: tamaño del array de bits.
# Salida: 
#             -array: Array de bits.
###############################################################################

def genRandomBitString(n_bits):
	return np.random.randint(2, size=n_bits)



###############################################################################
# Nombre: MH_encrypt
# Funcionalidad: Genera la lista con el cifrado de cada bloque de la lista de
# bits.
# Argumentos: 
#             -s: Array de bits a cifrar.
#             -lPub: Solución pública.
#             -mod: módulo necesario para el cifrado.
# Salida: 
#             -cifrado: lista con el cifrado de cada bloque.
###############################################################################

def MH_encrypt(s, lPub, mod):
	tamMod = len(s)%len(lPub)
	# Generar multiplicador e inverso adecuado a la sucesión lPub
	# mul = multiplier(mod, np.sum(lPub))
	# inv = inverse(mul, mod)

 # 	Generar solución pública asociada a la sucesión supercreciente lPub.
 # 	lPublica = genSucesionPublica(lPub, mul, mod)
 
	#Rellenamos con 0 para conseguir un tamaño multiplo de lPub.
	if tamMod != 0:
		zero = np.zeros((len(lPub)-tamMod,), dtype=np.int)
		s = np.concatenate((s, zero), axis=0)

	#Transforammos array en matriz MxN
	sArray = np.array(s)
	matrix = np.reshape(sArray, (-1, len(lPub)))
	print matrix

	#Transformamos cada bloque en su entero para poder cifrarlo.
	res = []
	for k in matrix:
		sum = 0
		for i in range(len(lPub)):
			if k[i] == 1:
				sum += lPub[i]
		res.append(sum)
		
	return res


###############################################################################
# Nombre: block_decrypt
# Funcionalidad: Genera el código descifrado de un bloque en una lista.
# Argumentos: 
#             -C: Cifrado de un bloque.
#             -l_sc: Lista privada.
#             -mod: Módulo necesario para el descifrado.
#			  -inv: Inverso del multiplicador y el módulo usado para el cifrado.
# Salida: 
#             -res: lista con el bloque descifrado.
###############################################################################

def block_decrypt(C, l_sc, inv, mod):
	#Cálculo de C'.
	cprim = inv * C % mod
	print cprim
	print C
	#Descifrado
	res = []
	w = len(l_sc)-1
	while w >= 0:
		if l_sc[w] > cprim:
			res.append(0)
		else:
			res.append(1)
			cprim = cprim - l_sc[w]
		w -= 1

	#Como el append introduce los números en orden inverso invertimos la lista antes de devolverlo.
	return res[::-1] 


###############################################################################
# Nombre: l_decrypt
# Funcionalidad: Genera el código descifrado de una lista de bloques cifrados.
# Argumentos: 
#             -l_cifra: Lista de bloques cifrados.
#             -l_sc: Lista privada.
#             -mod: Módulo necesario para el descifrado.
#			  -inv: Inverso del multiplicador y el módulo usado para el cifrado.
# Salida: 
#             -res: lista de los bloques descifrados concatenados.
###############################################################################

def l_decrypt(l_ifra, l_sc, inv, mod):
	res = []
	for k in l_ifra:
		res.extend(block_decrypt(k, l_sc, inv, mod))
	return res


#TESTS 
print "inverso"
print inverse(7, 50)
# lSC = genSuperCrec(10)
# print "lSC"
# print lSC
# print mcd(65, 15)
# print multiplier(30, 10)
# print inverse(117, 244)
# mod,mul,inv = modMultInv(lSC)
# print "mod"
# print mod
# print "mul"
# print mul
# print "inv"
# print inv
# print "solPublica"
# solPublica = genSucesionPublica(lSC, mul, mod)
# print solPublica
# print "solPrivada"
# solPrivada = lPub_2_lSC(solPublica, inv, mod)
# print solPrivada
# print "array de bits"
# bits = genRandomBitString(16)
# print bits
bits = [0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
mod = 50
mul = 7
lSC = [1,3,6,12,24]
inv = 43
print "solPublica"
solPublica = genSucesionPublica(lSC, mul, mod)
print solPublica
print "Cifrado"
res = MH_encrypt(bits, solPublica, mod)
print res
print "Descrifrado de bloque 1"
print res[0]
print block_decrypt(res[0], lSC, inv, mod)
print "Descrifrado completo"
print l_decrypt(res, lSC, inv, mod)