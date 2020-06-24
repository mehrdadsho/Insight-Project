from __future__ import absolute_import, print_function
import os
import pickle
import pandas as pd
import numpy as np

import ModelTraining_Classes

############################################################
#                        Define Inputs                     #
############################################################
Dataset_directory = '/Users/mehrdadshokrabadi/Documents/Miscellaneous/DS/Insight/Project/Data'
Save_directory = '/Users/mehrdadshokrabadi/Documents/Miscellaneous/DS/Insight/Project/Model Training'
<<<<<<< HEAD
=======
#Paths to pre-processing models and lexicons
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48

#Parameters of the models to be fitted to the data
model_type = 'RandomForest'
model_par = {"n_estimators" : 1250,
                "max_depth" : 10,
                "min_samples_leaf" : 20,
                "max_features" : 'sqrt',
                "class_weight" : 'balanced',
                "criterion" : 'entropy',
                "random_state" : 42,
                "n_jobs" : -1}

<<<<<<< HEAD
do_ModelTuning_on_RandomForest = False;
do_prediction_on_trainset_RandomForest = True;
do_prediction_on_testset_randomforest = True

tunning_params = {"min_samples_leaf" : [3,10,20],
         "n_estimators": np.arange(1000, 2000, 250),
        "max_depth":[10,20,30]}
model_par_not_for_tuning = {"random_state" : 42,
                "class_weight" : 'balanced',
                "max_features" : 'sqrt',
                "criterion" : 'entropy',
                "n_jobs"  : -1}
scoring_function = 'custom'

=======
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48
os.chdir(Dataset_directory)
train_cleaned = pickle.load( open( "train_cleaned.pkl", "rb" ) )
train_engineered = pickle.load( open( "train_engineered.pkl", "rb" ) )
train_sentimentAnalysis = pickle.load( open( "train_sentimentAnalysis.pkl", "rb" ) )
train_word2vec = pickle.load( open( "train_word2vec.pkl", "rb" ) )


##############################################################################
#                         Fitting a random forest model                      #
##############################################################################
<<<<<<< HEAD
x = train_cleaned
x = x.drop(x[x.Bias == 'Not Rated'].index)
x = x.drop(x[x.Bias == 'Mixed'].index)
train_cleaned.Bias, labels = pd.Categorical(pd.factorize(train_cleaned.Bias)[0]), pd.Categorical(pd.factorize(train_cleaned.Bias)[1])

ypred_RF = ModelTraining_Classes.RandomForest_Classes(train_word2vec[0], train_cleaned['Bias'], labels,
                                                            'RandomForest',
                                                      tunning_params, scoring_function, model_par_not_for_tuning,
                                                      do_ModelTuning_on_RandomForest, do_prediction_on_trainset_RandomForest,
                                                      do_prediction_on_testset_randomforest,
                                                      model_par, n_splits = 5, num_round = 500)
=======
train_cleaned.Bias, labels = pd.Categorical(pd.factorize(train_cleaned.Bias)[0]), pd.Categorical(pd.factorize(train_cleaned.Bias)[1])
ypred_RF = ModelTraining_Classes.RandomForest_Classes(train_word2vec[0], train_cleaned['Bias'],
                                                            'RandomForest', model_par, n_splits = 5, num_round = 500)
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48
ypred_RF = ypred_RF.Output

