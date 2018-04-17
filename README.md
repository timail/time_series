**СКАВРя** по методичке _В.В.Витязева_

Spectral Correlation Analysis of Time Series. 
(according to training manual by V.V.Vityazev)

---
В **main.py** есть константа **EXTRA_PLOTS**. 
Если ей выставить значение _True_, то программа строит два графика, не указанных в методичке --- график обрезанной по параметрю окна Тьюки N* коррелограммы и график взвешенной коррелограммы. 
---
**lib/**
- conf.py --- _параметры окна Тьюки_
- func.py --- _функция, генерирующая ряд_
- default_const.py --- _константы для ряда из примера_

---
Tested on **Python 3.6.4**
Lib dependences:
- numpy;
- matplotlib.


---
Астрономия СПбГУ, 2018.

III курс, шестой сем. 