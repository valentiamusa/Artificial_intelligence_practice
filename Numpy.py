import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns


sns.displot([0, 1, 2, 3, 6, 5])

#plt.show()

from numpy import random

 
def myadd(x, y):
  return x+y

myadd = np.frompyfunc(myadd, 2, 1)

print(myadd([1, 2, 3, 4], [5, 6, 7, 8]))
import math

print(math.gcd(84, 30))  # Output: 6
