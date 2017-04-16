__author__ = 'mike_bowles'
import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot
from math import exp

target_url = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/wine-quality/winequality-red.csv")
wine = pd.read_csv(target_url,header=0, sep=";")
wineCols = len(wine.columns)

#calculate correlation matrix
corMat = DataFrame(wine.corr())

#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()