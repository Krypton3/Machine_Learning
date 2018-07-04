from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def best_fit_slope_intercept(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) ** 2) - mean(xs * xs))
    b = mean(ys) - m * mean(xs)
    return m, b


XS = [1, 2, 3, 4, 5]
YS = [5, 4, 6, 5, 6]

xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

m, b = best_fit_slope_intercept(xs, ys)

regression_line = []
for x in xs:
    regression_line.append((m * x) + b)

predict_x = 7

predict_y = (m*predict_x) + b

print(predict_y)

plt.scatter(xs, ys, color='#003F72', label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
# print(m, b)
