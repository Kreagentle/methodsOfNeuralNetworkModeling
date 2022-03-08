# генерация файла txt

import numpy 
from numpy import *

kolvo = 22
# для генерации
TMAX = 1 # максимальное время
TAU = TMAX / (kolvo - 1)
VT = numpy.zeros(kolvo+1)
for K in range(1,kolvo+1):
    T = (K - 0.5) * TAU # время, за которое мы дошли до K-того узла
    VT[K] = (K - 0.5) * TAU
    if (T >= 0.6): 
        VT[K] = 0
    
# 1、случайно сгенерированные точки данных
data = mat(zeros((kolvo, 2)))
m = shape(data)[0]
x = numpy.linspace(0, 100, kolvo)
for i in range(m):
    data[i, 0] = x[i]
    data[i, 1] = VT[i]
# 2、сохраните данные точки в файл "data3"
f = open("data2.txt", "w")
m,n = shape(data)
for i in range(m):
    tmp =[]
    for j in range(n):
        tmp.append(str(data[i,j]))
    # tmp.append(str(1))
    f.write("\t".join(tmp) + "\n")
f.close()