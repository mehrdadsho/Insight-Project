from __future__ import absolute_import, print_function
import os
import pickle
import pandas as pd
import gensim
from gensim.test.utils import get_tmpfile

import EDA_DataCleaning_Classes

############################################################
#                        Define Inputs                     #
############################################################
<<<<<<< HEAD
Dataset_directory = '/Users/mehrdadshokrabadi/Documents/Miscellaneous/DS/Insight/Project/Data'
=======
Dataset_directory = '/Users/mehrdadshokrabadi/Documents/Miscellaneous/DS/Insight/Project/Web Scraping'
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48
Save_directory = '/Users/mehrdadshokrabadi/Documents/Miscellaneous/DS/Insight/Project/Data'
#Paths to pre-processing models and lexicons
genism_word2vec_path = "/Users/mehrdadshokrabadi/vectors.kv";

os.chdir(Dataset_directory)
train = pickle.load( open( "WebScrapingOutputs_NewsBodies.pkl", "rb" ) )
# train = pd.DataFrame(data=train)
<<<<<<< HEAD
train = train
=======
train = train.iloc[:5]
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48
text_fields = {'news_body'}

min_df = 10
max_features = 10000
list_of_ngrams = [(1,1),(2,2)]

##############################################################################
#                                 Data Cleaning                              #
##############################################################################
train_cleaned = []
for text_field in text_fields:
    CleanedData = EDA_DataCleaning_Classes.DataCleaning_Classes(train, text_field)
    train_cleaned.append(CleanedData.data)
train_cleaned = pd.concat(train_cleaned,ignore_index=True)
# train_cleaned = pd.DataFrame(train_cleaned)
# Saving the data structures
os.chdir(Save_directory)
f = open("train_cleaned.pkl","wb")
pickle.dump(train_cleaned,f)
f.close()


##############################################################################
#                             Feature Engineering                            #
##############################################################################
train_engineered = []
columns = []
for text_field in text_fields:
    EngineeredData = EDA_DataCleaning_Classes.FeatureEngineering_Classes(train, text_field, min_df, max_features, list_of_ngrams)
    dic_keys = list(EngineeredData.Output.keys())
    for i,dic_key in enumerate(dic_keys):
        if i < 4:
            continue
        else:
            columns.append(dic_key)
            train_engineered.append(EngineeredData.Output[dic_key])
# train_engineered = pd.DataFrame(train_engineered)
# Saving the data structures
os.chdir(Save_directory)
f = open("train_engineered.pkl","wb")
pickle.dump(train_engineered,f)
f.close()

#############################################################################
#                             Sentiment Analysis                            #
#############################################################################
train_sentimentAnalysis = []
columns = []
for text_field in text_fields:
    SentimentAnalysis = EDA_DataCleaning_Classes.SentimentAnalysis_Classes(train, text_field)
    dic_keys = list(SentimentAnalysis.Output.keys())
    for i,dic_key in enumerate(dic_keys):
        columns.append(dic_key)
        train_sentimentAnalysis.append(SentimentAnalysis.Output[dic_key])

# train_sentimentAnalysis = pd.DataFrame(train_sentimentAnalysis, columns = columns)
# Saving the data structures
os.chdir(Save_directory)
f = open("train_sentimentAnalysis.pkl","wb")
pickle.dump(train_sentimentAnalysis,f)
f.close()

#############################################################################
#                             word2vec Analysis                             #
#############################################################################
fname = get_tmpfile(genism_word2vec_path)
word2vec = gensim.models.KeyedVectors.load(fname, mmap='r')
train_word2vec = []
columns = []
for text_field in text_fields:
    new_field = text_field + "_tokenized"
    word2vec_Analysis = EDA_DataCleaning_Classes.word2vec_analysis(word2vec, train_cleaned, new_field)
    train_word2vec.append(word2vec_Analysis.Output)
# train_word2vec = pd.DataFrame(train_word2vec)
# Saving the data structures
os.chdir(Save_directory)
f = open("train_word2vec.pkl","wb")
pickle.dump(train_word2vec,f)
f.close()
