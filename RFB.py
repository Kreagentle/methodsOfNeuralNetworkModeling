import numpy as np
import matplotlib. pyplot as plt
from datetime import datetime

class RBF(object):

    def __init__(self, hidden_shape, sigma=1.0):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self, X): # метод интерполяции
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def fit(self, X, Y):
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions
    

DELTA=0.001 # уровень погрешности во входных данных
N=100 # число узлов сетки по пространству 
M=100 # число узлов сетки по времени

XL = 0 # левый конец отрезка
XR = 1 # правый конец отрезка
XD = 0.3 # точка наблюдения
TMAX = 1 # максимальное время 

# generating data
x = np.linspace(0, 1, M+1)

VT = np.zeros(M+1) # точная зависимость правой части от времени

# сетка (посмотреть как тут работает)
H = (XR - XL) / (N - 1)
TAU = TMAX / (M - 1)
ND = int(1 + (XD + 0.5*H) / H) # количество шагов до точки налюдения

# ПРЯМАЯ ЗАДАЧА

# источник (нормально работает)
for K in range(1,M+1):
        T = (K - 0.5) * TAU # время, за которое мы дошли до K-того узла
        VT[K] = (K - 0.5) * TAU
        if (T >= 0.6): 
            VT[K] = 0
            
# fitting RBF-Network with data
model = RBF(hidden_shape=10, sigma=1.)
start = datetime.now()
for i in range(50):
    model.fit(x, VT)
print(datetime.now() - start)
y_pred = model.predict(x)
score = np.mean((VT-y_pred)**2) # ошибка модели (сравниваем тестовые значения и модель) сделать с другим x и VT
# print(score)

# print(y_pred)
# сравнение средней ошибки
# plotting 1D interpolation
plt.plot(x, VT, 'b-', label='точное')
plt.plot(x, y_pred, 'r-', label='приближенное')
plt.legend(loc='upper right')
# plt.title('Interpolation using a RBFN')
plt.show()