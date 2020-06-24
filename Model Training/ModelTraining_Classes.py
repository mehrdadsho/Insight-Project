import warnings
warnings.filterwarnings('ignore')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#### import packages used in EDA ####
import seaborn as sns
<<<<<<< HEAD

import sklearn

import matplotlib.pyplot as plt

import timeit



class RandomForest_Classes(object):
    def __init__(self, train_x, train_y, labels, model_type,
                 tunning_params, scoring_function, model_par_tuning,
                 do_ModelTuning_on_RandomForest, do_prediction_on_trainset_RandomForest,
                 do_prediction_on_testset_randomforest,
                 model_par, n_splits, num_round):
        #Initialization of the data structure
        self.Output = {}
        self.do_ModelTuning_on_RandomForest = do_ModelTuning_on_RandomForest
        self.do_prediction_on_trainset_RandomForest = do_prediction_on_trainset_RandomForest
        self.do_prediction_on_testset_randomforest = do_prediction_on_testset_randomforest
        #Methods to fill in the data structure
        # self.Add_ClassTags(train_x, dataset_with_bias_labels)
        self.Perform_HyperParameter_Tuning(train_x, train_y, tunning_params, model_type, scoring_function, model_par_tuning)
        self.make_predictions_on_trainset_RandomForest(train_x, train_y, model_type, model_par, n_splits, num_round, labels)

    def Perform_HyperParameter_Tuning(self, dataset, y, tunning_params, model_type, scoring_function, model_par):
        """
        ##### Glossary of the function inputs #####
        test_size: size of the validation set
        tunning_params: parameters that grid search would perform model tuning for
        model_type:'RandomForest'
        scoring_function: could either be a custom scoring function defined as "custom" or one of the sklearn's default scorung functions
        model_par: the fixed parameters of the model
        """
        # First divide the trainset into train and validation sets
        if self.do_ModelTuning_on_RandomForest:
            (x_train, x_test, y_train, y_test) = sklearn.model_selection.train_test_split(dataset, y, test_size=0.25,
                                                                                         random_state=42, stratify = y,
                                                                                         shuffle=True)
            start = timeit.default_timer()
            # Define a custom scoring function to select the best subset of the tuning parameters
            def my_custom_loss_func(y_true, y_pred):
                score1 = sklearn.metrics.fbeta_score(y_true, y_pred,  average='weighted', beta=1.0)
                return score1
            def f1_score_func(y_true, y_pred):
                score = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')
                return score
            if scoring_function == 'custom':
                scorer = sklearn.metrics.make_scorer(my_custom_loss_func, greater_is_better=True)
            elif scoring_function == 'f1_score':
                scorer = f1_score_func

            # Define the model to be tuned
            if model_type == 'RandomForest':
                model = sklearn.ensemble.RandomForestClassifier(**model_par)

            # Perform the grid search and print out the best parameters
            grid = sklearn.model_selection.GridSearchCV(model, tunning_params,
                                                        scoring = scorer, verbose = 2)
            grid.fit(x_train, y_train)
            acc = grid.score(x_test, y_test)
            print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
            print("[INFO] randomized search best parameters: {}".format(grid.best_params_))

            stop = timeit.default_timer()
            print('Time to completion (minutes): ', (stop - start)/60.0)
        else:
            pass


    ########## Fit a random forest model to the trainset ##########
    def make_predictions_on_trainset_RandomForest(self, x, y, model_type, model_par, n_splits, num_round, labels):
=======
import re
import string
import nltk
#nltk.download()
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))
color = sns.color_palette()

from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import sklearn

### packages for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

tokenizer=TweetTokenizer()
lem = WordNetLemmatizer()


class RandomForest_Classes(object):
    def __init__(self, train_x, train_y, model_type, model_par, n_splits, num_round):
        #Initialization of the data structure
        self.Output = {}
        #Methods to fill in the data structure
        self.make_predictions_on_trainset_RandomForest(train_x, train_y, model_type, model_par, n_splits, num_round)
        self.Perform_model_evaluation(self, train_y)

    ########## Fit a random forest model to the trainset ##########
    def make_predictions_on_trainset_RandomForest(self, x, y, model_type, model_par, n_splits, num_round):
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48
        """
        ##### Glossary of the function inputs #####
        x: features
        y: class labels
        model_type: 'RandomForest'
        model_par: the fixed parameters of the model
        n_splits = number of train-validation splits
        """
<<<<<<< HEAD
        if self.do_prediction_on_trainset_RandomForest:
            yoof = np.zeros((len(x),1))
            kf = sklearn.model_selection.StratifiedKFold(n_splits, random_state=42, shuffle=True)
            fold = 0
            for in_index, oof_index in kf.split(x,y):
                fold += 1
                print(f'fold {fold} of {n_splits}')
                X_in, X_oof = x.values[in_index], x.values[oof_index]
                y_in, y_out = y.values[in_index], y.values[oof_index]
                if model_type == 'RandomForest':
                    model = sklearn.ensemble.RandomForestClassifier(**model_par)
                    model.fit(X_in, y_in)
                    yoof[oof_index,0] = model.predict(X_oof)
            # Print out the 10 most important features
            importances = model.feature_importances_
            features = x.columns
            importances = model.feature_importances_
            indices = np.argsort(importances)
            indices = indices[-min(10,len(features)):]
            plt.title('10 Most Important Features')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.show()
            self.Output['yoof'] = yoof
            self.Perform_model_evaluation(y, labels)
        else:
            pass

    def Perform_model_evaluation(self, y_true, labels):
=======
        yoof = np.zeros((len(x),1))
        kf = sklearn.model_selection.StratifiedKFold(n_splits, random_state=42, shuffle=True)
        fold = 0
        for in_index, oof_index in kf.split(x,y):
            fold += 1
            print(f'fold {fold} of {n_splits}')
            X_in, X_oof = x.values[in_index], x.values[oof_index]
            y_in, y_out = y.values[in_index], y.values[oof_index]

            if model_type == 'RandomForest':
                model = sklearn.ensemble.RandomForestClassifier(**model_par)
                model.fit(X_in, y_in)
                yoof[oof_index,0] = model.predict(X_oof)
        self.Output['yoof'] = yoof

    def Perform_model_evaluation(self, y_true):
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48
        print("The overall accuracy of the prediction is %.2f%%" % (sklearn.metrics.accuracy_score(y_true, self.Output['yoof'] )*100))
        print('\n'.join('The recall rate for Class {} is: {:.2f}%'.format(*k) for k in enumerate(sklearn.metrics.recall_score(y_true, self.Output['yoof'] , average=None)*100)))
        print('\n'.join('The precision rate for Class {} is: {:.2f}%'.format(*k) for k in enumerate(sklearn.metrics.precision_score(y_true, self.Output['yoof'] , average=None)*100)))
        print('\n'.join('The f1 score rate for Class {} is: {:.2f}%'.format(*k) for k in enumerate(sklearn.metrics.f1_score(y_true, self.Output['yoof'] , average=None)*100)))
        ## plot the confusion matrix
<<<<<<< HEAD
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, self.Output['yoof'])
        bp = sns.heatmap(confusion_matrix, annot=True)
        bp.set_xticklabels(labels)
        bp.set_yticklabels(labels)
        bp.tick_params(labelsize=20)
        bp.set_xlabel("True Label",fontsize=30)
        bp.set_ylabel("Predicted Label",fontsize=30)
        fig = plt.gcf()
        fig.set_size_inches( 14, 10)

    def make_predictions_on_testset(self, x_train, y_train,  model_type, model_par):
        """
        ##### Glossary of the function inputs #####
        x: features
        y: class labels
        model_type:  'RandomForest'
        scoring_function: could either be a custom scoring function defined as "scorer" or one of the sklearn's default scorung functions
        model_par: the fixed parameters of the model
        """
        if self.do_prediction_on_testset_randomforest:
            if model_type == 'RandomForest':
                model = sklearn.ensemble.RandomForestClassifier(**model_par)
                model.fit(x_train, y_train)
                self.Output['Final_trained_model'] = model
            else:
                pass
=======
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, self.Output['yoof'] )
        sns.heatmap(confusion_matrix, annot=True)
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48



