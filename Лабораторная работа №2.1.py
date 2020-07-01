#здесь происходит опрделение гласных и согласных букв английского алфавита
import numpy as np
import matplotlib.pyplot as plt
import os

n = os.listdir(r'C:\Users\User\Desktop\alpha')  # список файлов в папке (можно указать формат, но в данном случае в папке только файлы .png)

w = np.zeros(168)  # веса

# обучающий вектор для нейрона (работаю с трехмерным массивом)
D = []
for i in n:
    D.append(plt.imread(i))


# истинные значения для нейрона
def ans(listname): # функция для генерации истинных значений
    sogl = ('AEIOUY')
    Y = []
    for i in listname:
        if sogl.find(i.split('.png')[0]) == -1:# проверка на наличие то или иной буквы в строке с гласными
            Y.append(1)
        else:
            Y.append(0)
    return Y
Y = ans(n)  # 0 - гласная, 1 - согласная

α = 0.2
β = -0.4
σ = lambda x: 1 if x > 0 else 0 #функця активации


def f(x):
    s = β + np.sum(x * w)
    return σ(s)


def train():
    global w
    _w = w.copy()
    for x, y in zip(D, Y):  # для каждого файла(буквы) и каждого истинного значения
        x = x.flatten()
        w += α * (y - f(x))*x # меняются весы
    return (w != _w).any()


while train():
    print(w)

cit = (f(plt.imread('D.png').flatten()))
print(cit)
if cit == 1:
    print('Cогласная')
else:
    print('Гласная')