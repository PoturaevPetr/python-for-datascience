import numpy as np
import matplotlib.pyplot as plt
import os

n = os.listdir(
    r'C:\Users\User\Desktop\alpha')  # список файлов в папке (можно указать формат, но в данном случае в папке только файлы .png)

w = np.zeros(168).reshape(8, 7, 3)  # веса

# обучающий вектор для нейрона
D = []
for i in n:
    D.append(plt.imread(i))

# истинные значения для нейрона
Y = np.array(
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1])  # 0 - гласная, 1 - согласная
print(Y)
α = 0.2
β = -0.4
σ = lambda x: 1 if x > 0 else 0


def f(x):
    s = β + np.sum(x * w)

    if σ(s) == 1:
        print('Cогласная')
    else:
        print('Гласная')
    return σ(s)


def train():
    global w
    _w = w.copy()
    for x, y in zip(D, Y):  # для каждого файла(буквы) и каждого истинного значения
        w += α * (y - f(x)) * x
        print(w)
    return (w != _w).any()


while train():
    print(w)

###

print(f(plt.imread('E.png')))  # для удобства, вохдной сигнал берертся из директории.
