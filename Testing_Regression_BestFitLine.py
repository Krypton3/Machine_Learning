from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')

# finding slope and y intercept b to find the best fit line


def create_dataset(hm, variance, step = 2, correlation = False):
    val = 1
    ys = [] # labels testing
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))] # features testing
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) ** 2) - mean(xs ** 2)))

    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error_calc(ys_main, ys_line):
    return sum((ys_line-ys_main)**2)


def coefficient_of_determination(ys_main, ys_line):
    y_mean_line = [mean(ys_main) for y in ys_main]
    squared_error_regr = squared_error_calc(ys_main, ys_line)
    squared_error_y_mean = squared_error_calc(ys_main, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


xs, ys = create_dataset(40,10,2,correlation=False)
m, b = best_fit_slope(xs, ys)
regression_fit_line = [m*x+b for x in xs]
r_squared = coefficient_of_determination(ys, regression_fit_line)
print(r_squared)

plt.scatter(xs, ys, label='data', color='#003F72')
plt.plot(xs, regression_fit_line, label='Regression Line')
plt.legend(loc=4)
plt.show()

