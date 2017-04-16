__author__ = 'mike_bowles'
import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot
from math import exp

target_url = ("https://archive.ics.uci.edu/ml/machine-"
              "learning-databases/glass/glass.data")
glass = pd.read_csv(target_url,header=None, prefix="V")
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si',
                 'K', 'Ca', 'Ba', 'Fe', 'Type']
ncols = len(glass.columns)

#calculate correlation matrix
corMat = DataFrame(glass.iloc[:, 1:(ncols - 1)].corr())

#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()