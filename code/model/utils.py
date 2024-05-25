import numpy as np
from sklearn import metrics

# Predictive models
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor


def supervised_model_training(x_train, y_train, x_test, 
                              y_test, model_name):
     
  if model_name == 'logisticregression':
    model  = LogisticRegression()
  elif model_name == 'randomforest':      
    model = RandomForestClassifier()
  elif model_name == 'gaussiannb':  
    model = GaussianNB()
  elif model_name == 'bernoullinb':  
    model        = BernoulliNB()
  elif model_name == 'multinb':  
    model        = MultinomialNB()
  elif model_name == 'svmlin':         
    model        = svm.LinearSVC() 
  elif model_name == 'gbm':         
    model         = GradientBoostingClassifier()   
  elif model_name == 'Extra Trees':
    model =  ExtraTreesClassifier(n_estimators=20)
  elif model_name == 'LDA':
    model =  LinearDiscriminantAnalysis() 
  elif model_name == 'Passive Aggressive':
    model =   PassiveAggressiveClassifier()
  elif model_name == 'AdaBoost':
    model =   AdaBoostClassifier()
  elif model_name == 'Bagging':
    model =   BaggingClassifier()
  elif model_name == 'xgb':
    model =   XGBRegressor()                                
  
  if(model_name=='svmlin' or model_name=='Passive Aggressive'): 
    model.fit(x_train, y_train)
    predict = model.decision_function(x_test)
  elif (model_name =='xgb'):
    model.fit(np.asarray(x_train), y_train)
    predict = model.predict(np.asarray(x_test))
  else:
    model.fit(x_train, y_train)
    predict = model.predict_proba(x_test)[:,1]
        
  # AUC / AUPRC Computation
  auc = metrics.roc_auc_score(y_test, predict)
  apr = metrics.average_precision_score(y_test, predict)
  
  return auc, apr    