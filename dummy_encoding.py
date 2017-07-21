import pandas
from pandas.tools.plotting import scatter_matrix
import csv
import numpy 
from sklearn import metrics


urlx = "cluster_dataset.csv"
#urly = "y.csv"

namesx = ["Score","Cluster","Age","INCOME","WEALTH1","MINRAMNT","MAXRAMNT",
          "LASTGIFT","Period","WEALTH2","NGIFTALL","TAMT"]

#namesy = ["amt"]
#f = csv.reader(open("dsf.csv","rb"))
#w = csv.writer(open("op2.csv","ab"))
#w.writerow(['income','wealth','age','status','score',"gender","state","predicted","cluster"])
#Y = pandas.read_csv(urly, names = namesy)

X = pandas.read_csv(urlx, names = namesx)

print "......................Converting-dummy encoding........................"

dummy_wealth1=pandas.get_dummies(X['WEALTH1'],prefix='Wealth1') 
dummy_score=pandas.get_dummies(X['Score'],prefix='Score')         
dummy_cluster=pandas.get_dummies(X['Cluster'],prefix='Cluster')         
dummy_age=pandas.get_dummies(X['Age'],prefix='Age')         
dummy_income=pandas.get_dummies(X['INCOME'],prefix='Income')                
dummy_wealth2=pandas.get_dummies(X['WEALTH2'],prefix='Wealth2')         

col_to_keep=['MINRAMNT',"MAXRAMNT","LASTGIFT","Period","NGIFTALL","TAMT"]

# X_new is neew dataframe

X_new=X[col_to_keep].join(dummy_score.ix[:,'Score_3':])
X_new=X_new.join(dummy_wealth1.ix[:,'Wealth1_2':])
X_new=X_new.join(dummy_cluster.ix[:,'Cluster_2':])
X_new=X_new.join(dummy_age.ix[:,'Age_2':])
X_new=X_new.join(dummy_income.ix[:,'Income_2':])
X_new=X_new.join(dummy_wealth2.ix[:,'Wealth2_2':])


#print X_new
#X_new.to_csv('op.csv', sep='\t', encoding='utf-8')



#saving to csv
X_new.to_csv('op.csv')


print "done"
