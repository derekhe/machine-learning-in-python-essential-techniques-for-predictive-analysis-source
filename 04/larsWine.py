__author__ = 'mike_bowles'
import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot
from math import fabs

target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine = pd.read_csv(target_url,header=0, sep=";")

#normalize the wine data
summary = wine.describe()
print(summary)

wineNormalized = wine
ncols = len(wineNormalized.columns)
nrows = len(wineNormalized)

for i in range(ncols):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    wineNormalized.iloc[:,i:(i + 1)] = (wineNormalized.iloc[:,i:(i + 1)] - mean) / sd

#initialize a vector of coefficients beta
beta = [0.0] * (ncols - 1)
#initialize matrix of betas at each step
betaMat = []
betaMat.append(list(beta))
#initialize residuals list
residuals = [0.0] * nrows

#number of steps to take
nSteps = 100
stepSize = 0.1

for i in range(nSteps):
    #calculate residuals
    for j in range(nrows):
        residuals[j] = wineNormalized.iloc[j, (ncols - 1)]
        for k in range(ncols - 1):
            residuals[j] += - wineNormalized.iloc[j, k] * beta[k]

    #calculate correlation between attribute columns from normalized wine and residual
    corr = [0.0] * (ncols - 1)

    for j in range(ncols - 1):
        for k in range(nrows):
            corr[j] += wineNormalized.iloc[k,j] * residuals[k] / nrows

    iStar = 0
    corrStar = corr[0]

    for j in range(1, (ncols - 1)):
        if abs(corrStar) < abs(corr[j]):
            iStar = j; corrStar = corr[j]

    beta[iStar] += stepSize * corrStar / abs(corrStar)
    betaMat.append(list(beta))


for i in range(ncols - 1):
    #plot range of beta values for each attribute
    coefCurve = betaMat[0:nSteps][i]
    coefCurve.plot()

plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()
