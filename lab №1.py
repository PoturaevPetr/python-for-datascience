import numpy as np #импортирую модуль numpy

M = np.genfromtxt('test.csv', delimiter=';') # Считывание матрицы из файла test.csv, разделитель - ;
print(M) # показываю матрицу
minimum = np.min(M) # Определение минимама матрицы
maximum = np.max(M) # Определение максимума матрицы
print('min = ' , minimum, '\n', 'max = ', maximum) # Показываю значения минимама и максимума

r1 = np.where(M == minimum) # Нахожу координаты минимума
r2 = np.where(M == maximum) # нахожу координаты максимума
M[r1], M[r2] = M[r2], M[r1] # меняю их местами
print(M)