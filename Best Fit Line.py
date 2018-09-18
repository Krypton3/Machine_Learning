from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# finding slope and y intercept b to find the best fit line


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


xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

m, b = best_fit_slope(xs, ys)

print(m, b)

regression_fit_line = []

for x in xs:
    regression_fit_line.append(m*x+b)

# plt.scatter(xs, ys, color='#003F72')
# plt.plot(xs, regression_fit_line)
# plt.show()


predict_x = 7
predict_y = (m*predict_x)+b

plt.scatter(xs, ys, label='data', color='#003F72')
plt.scatter(predict_x, predict_y, color='g', label='Prediction')
plt.plot(xs, regression_fit_line, label='Regression Line')
plt.legend(loc=4)
plt.show()

r_squared = coefficient_of_determination(ys, regression_fit_line)
print(r_squared)