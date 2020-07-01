# здесь происходит выделение только гласных букв и определяется какая именно буква на картинке и определения качества распознавания
import numpy as np
import matplotlib.pyplot as plt
import os

image = os.listdir(r'C:\Users\User\Desktop\alpha')  # список файлов в папке (можно указать формат, но в данном случае в папке только файлы .png)


def ans(listname):  # для определения только гласных букв из всех имеющихся
    glas = ('AEIOUY')
    S = []
    for i in listname:
        if glas.find(i.split('.png')[0]) != -1:  # проверка на наличие то или иной буквы в строке с гласными
            S.append(i)
    return S


n = ans(image)  # только гласные буквы
print(n)
# обучающий вектор для нейрона
D = []
for i in range(6):  # для каждой буквы (знаем количество гласный - 6)
    x = (np.dot(plt.imread(n[i])[..., :3], [1, 1, 1]) > 1).astype(int).flatten()
    y = (np.array(list(bin(i)[2:].zfill(3))) == '1').astype(int)
    D += [[x, y]]

w = np.zeros((D[0][0].shape[0], D[0][1].shape[0]))  # веса

β = -0.4

α = 0.2

σ = lambda x: (x > 1).astype(int)  # функция активации


def f(x):  #
    s = β + x @ w
    return σ(s)


def train():  # обучаюшая функция, происходит изменение весов
    global w
    _w = w.copy()
    for x, y in D:
        i = np.where(x > 0)  # получем индекс ненулевого элемента
        w[i] += α * (y - f(x))  # меняеются только те веса где х больше нуля
    return (w != _w).any()


while train():
    print(w)

# значения σ(s) для кадой буквы
book = np.array([np.array([0, 0, 0]),
                 np.array([0, 0, 1]),
                 np.array([0, 1, 0]),
                 np.array([0, 1, 1]),
                 np.array([1, 0, 0]),
                 np.array([1, 0, 1])])
df = ['A', 'E', 'I', 'O', 'U', 'Y']
et = []
for i in range(6):  # для каждой гласной буквы
    cit = (f((np.dot(plt.imread(n[i])[..., :3], [1, 1, 1]) > 1).astype(int).flatten()))
    print(cit)
    et += [cit]
    if ((book[i] == cit).all()) == True:  # проверем какому значению из book соответствует σ(s)
        print('Гласная буква - ', df[i])  # получаем нужную букву

### определение качества распознавания
print('\nКачество распознавания')
from scipy import ndimage

R = []
theta = [10, 20, 30, 40, 50, 60, 70, 80]
div = []
for th in theta:
    K = []
    print('\nПоворот на ' + str(th) + ' градусов')
    procent = 0
    for i in range(6):
        rotated = ndimage.rotate((np.dot(plt.imread(n[i])[..., :3], [1, 1, 1])), th, reshape=0)
        x = (rotated > 1).astype(int).flatten()
        cit = (f(x))
        print(cit)
        K += [cit]

        if ((book[i] == cit).all()) == True:  # проверем какому значению из book соответствует σ(s)
            print('Гласная буква - ', df[i])  # получаем нужную букву
            procent += 1
    div += [procent / 6]
plt.plot(theta, div)
plt.xlabel(r'$\theta$')
plt.ylabel('Качество распознавания')