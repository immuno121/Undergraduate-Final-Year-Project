
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import csv
import numpy 
from pandas import DataFrame, Series
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
import scipy, scipy.stats

from sklearn import metrics
from sklearn.metrics import pairwise_distances
from operator import itemgetter
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from scipy import stats

print "..............Fetchinga and Reading datasets............."
urlx = "data_1.csv"
urly = "y.csv"

namesx = ["MINRAMNT","MAXRAMNT","LASTGIFT","Period","NGIFTALL","TAMT","Score_3","Score_4","Score_5","Score_6","Score_7","Score_8","Score_9",
          "Score_10","Score_11","Score_12" ,"Wealth_2","Wealth_3","Wealth_4","Wealth_5","Wealth_6","Wealth_7","Wealth_8","Wealth_9","Cluster_2"
          ,"Cluster_3","Cluster_4","Cluster_5","Cluster_6","Cluster_7","Cluster_8","Cluster_9"
          ,"Cluster_10","Cluster_11","Age_2","Age_3","Age_4","Age_5","Age_6","Age_7","Age_8","Age_9"
          ,"Age_10","Income_2","Income_3","Income_4","Income_5","Income_6","Income_7"
           ,"Wealth2_2","Wealth2_3","Wealth2_4","Wealth2_5","Wealth2_6","Wealth2_7","Wealth2_8","Wealth2_9"]

namesy = ["amt"]

X = pandas.read_csv(urlx,names=namesx)
Y = pandas.read_csv(urly,names=namesy)
#mainDataset=numpy.array(X)
print "..............Calculating coefficients............."


# training & testing
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1)

reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)


print reg.coef_


output=reg.predict(X_test)

#print output





print("Mean squared error: %.2f"
      % mean_squared_error(Y_test,output))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % reg.score(X_test, Y_test))


results = sm.OLS(Y_train, X_train).fit()
print(results.summary())






# Plot outputs
##plt.plot(X, reg.predict(X), color='blue',
        # linewidth=3)
#
#plt.xticks(())
##
#plt.show()


#print "_____________________________________________________________________"
#print "................................Legen................................"
#RegPrediction=numpy.array(reg.predict(X2))
#print RegPrediction



#reg = linear_model.LinearRegression()

#reg.fit (X1,Y1)



#print "Linear"
#print reg.coef_

#output= reg.predict(X2)

#print output
    


#reg = linear_model.Ridge (alpha = .5)
#reg.fit (X,Y) 


#print reg.coef_

#print reg.intercept_ 
#reg = linear_model.Lasso(alpha = 0.1)
#reg.fit(X,Y
#print reg.coef_ 

#output=reg.predict(X)
##print output




























