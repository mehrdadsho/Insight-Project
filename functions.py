#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:47:55 2020

@author: mehrdadshokrabadi
"""

"""
Loading necessary packages
"""
#############################################################################
import re
import pandas as pd
import functions
import numpy as np

import newspaper
#newspaper3k
from newspaper import Article
from googlesearch import search

import nltk
nltk.data.path.append(r"/home/ubuntu/StreamlitApp/NLTKdata")
from nltk.tokenize import TweetTokenizer   
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
tokenizer=TweetTokenizer()
lem = WordNetLemmatizer()
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

from resources.feature_functions import Functions
Functions = Functions()

import streamlit as st

### packages for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

#############################################################################
"""
Functions for pulling article info and do Google search
"""
#############################################################################
def return_websearch_outputs(search_term, start_google_search, end_google_search):  
    query = search_term
    search_results_list = []
    for result in search(str(query),        # The query you want to run
                    lang = 'en',  # The language
                    num = 10,     # Number of results per page
                    start = start_google_search,    # First result to retrieve
                    stop = end_google_search,  # Last result to retrieve
                    pause = 0.0001,  # Lapse between HTTP requests
                   ):
        search_results_list.append(result)  
    return search_results_list

def search_each_source(input_URL):  
    input_dataframe = dict();
    article = Article(input_URL)
    article.download();
    try:
        article.parse();
        input_dataframe['news_body'] = article.text
        input_dataframe['NewsTitle'] = article.title
        source = re.split("[.]",input_URL)[1]
        input_dataframe['source'] = source  
        return input_dataframe
    except newspaper.ArticleException:
        print("ArticleException error")
        pass
  
#############################################################################
"""
Functions for cleaning up and standardizing text data
"""
#############################################################################
def call_data_cleaning_functions(input_dataframe, text_field):  
    input_dataframe = standardize_text(input_dataframe, text_field)
    clean_corpus = input_dataframe[text_field].apply(clean_text)
    input_dataframe[text_field] = clean_corpus
    input_dataframe = tokenizer_text(input_dataframe, text_field)  
    return input_dataframe    
########## Standardize the Data ##########
def standardize_text(df, text_field):
    df[text_field] = df[text_field].astype(str).str.replace(r"\\n", " ")
    df[text_field] = df[text_field].astype(str).str.replace(r"http\S+", "")
    df[text_field] = df[text_field].astype(str).str.replace(r"[$(),!?@\-\"\:\'\`\_\’\”\“\—\/\\]", " ")
    df[text_field] = df[text_field].astype(str).str.replace(r"[0-9]", " ")
    df[text_field] = df[text_field].astype(str).str.replace(r"http", "")
    df[text_field] = df[text_field].astype(str).str.replace(r"@\S+", "")
    df[text_field] = df[text_field].astype(str).str.replace(r"@", "at")
    df[text_field] = df[text_field].astype(str).str.lower()
    df[text_field] = df[text_field].astype(str).str.replace(r"\...", " ")
    return df
########## Tokenizing the text data ##########
def tokenizer_text(df, text_field):
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    new_field = text_field + "_tokenized"
    df[new_field] = df[text_field].apply(tokenizer.tokenize)
    all_words = [word for tokens in df[new_field] for word in tokens]
    sentence_lengths = [len(tokens) for tokens in df[new_field]]
    VOCAB = sorted(list(set(all_words)))
    # print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    # print("Max sentence length is %s" % max(sentence_lengths))
    return df
########## Word category ##########
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None
########## Clean up the Data ##########
def clean_text(data):
    """
    This function receives comments and returns clean word-list
    """
    #remove \n
    data = re.sub("\\n","",data)
    # remove leaky elements like ip,user
    data = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",data)
    #removing usernames
    data = re.sub("\[\[.*\]","",data)
    
    #Split the sentences into words
    words = tokenizer.tokenize(data)
    tagged = nltk.pos_tag(words)
    tags_treebank_tag = [get_wordnet_pos(w[1]) for w in tagged]
    
    for i, word in enumerate(words):
        if tags_treebank_tag[i] is None:
            words[i] = lem.lemmatize(word)
        else:
            words[i] = lem.lemmatize(word, tags_treebank_tag[i]) 
    # Removing stopwords
    words = [w for w in words if not w in eng_stopwords]     
    clean_sent = " ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    return(clean_sent)

#############################################################################
"""
Function for getting the key words in an article
"""
#############################################################################
def get_key_words(count_vectorizer,tfidf_vectorizer, text):
    text = [text]
    features = count_vectorizer.get_feature_names()
    X_test_counts = count_vectorizer.transform(text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_counts)
    cc = X_test_tfidf.toarray();
    DD = np.argsort(-1*cc)
    important_features = [features[DD[0][i]] for i in range(min(8,len(DD[DD > 0])))]
    return important_features


#############################################################################
"""
Functions for sentiment analysis
"""
#############################################################################
########## Do a sentiment analysis ##########
def sentiment_analyzer_scores_vader(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    return score 
def sentiment_analyzer_scores_TextBlob(sentence):
    """
    The polarity score is a float within the range [-1.0, 1.0].
    The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    """
    text = TextBlob(sentence)
    score = text.sentiment
    return score 
def feature_engineering(train, text_field):  
    TextBlob_scores = train[text_field].apply(sentiment_analyzer_scores_TextBlob)
    TextBlob_scores_subjectivity = [TextBlob_scores[i][1] for i in range(len(TextBlob_scores))]
    new_field = 'TextBlob_scores_subjectivity_' + text_field 
    train[new_field] = TextBlob_scores_subjectivity
    VADER_scores = train[text_field].apply(sentiment_analyzer_scores_vader)
    VADER_scores_compund = [VADER_scores[i]['compound'] for i in range(len(VADER_scores))]
    new_field = 'VADER_scores_compund_' + text_field 
    train[new_field] = VADER_scores_compund   
    return train


########## turn the source of the news to one-hot encoder ##########
def add_one_hot_encoder(input_dataframe, one_hot_column, one_hot_encoder): 
    dummy = np.expand_dims(input_dataframe[one_hot_column],axis=1)
    one_hot_source = one_hot_encoder.transform(dummy)
    one_hot_source = pd.DataFrame(one_hot_source)
    return one_hot_source

#############################################################################
"""
Functions for a more comprehensive sentiment analysis
"""
#############################################################################
def call_comprehensive_sentiment_analysis(input_dataframe, text_field):  
    cat_dict, stem_dict, counts_dict = Functions.load_LIWC_dictionaries()
    liwc_cats = [cat_dict[cat] for cat in cat_dict]
    pos_tags = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","WP$","WRB","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP"]
    # seq = ("Happiness, HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral, bias_count, assertives_count, factives_count, hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count, TTR,vad_neg,vad_neu,vad_pos,FKE,SMOG,stop,wordlen,WC,NB_pobj,NB_psubj,quotes,Exclaim,AllPunc,allcaps",",".join(pos_tags),",".join(liwc_cats))
    seq = ["Happiness","HarmVirtue","HarmVice","FairnessVirtue","FairnessVice","IngroupVirtue","IngroupVice","AuthorityVirtue","AuthorityVice","PurityVirtue","PurityVice","MoralityGeneral","bias_count","assertives_count","factives_count","hedges_count","implicatives_count","report_verbs_count","positive_op_count","negative_op_count","wneg_count","wpos_count","wneu_count","sneg_count","spos_count","sneu_count","TTR","vad_neg","vad_neu","vad_pos","FKE","SMOG","stop","wordlen","WC","NB_pobj","NB_psubj","quotes","Exclaim","AllPunc","allcaps"]
    seq2 = seq + pos_tags + liwc_cats
  
    sentiment_scores = []
    for i in range(input_dataframe.shape[0]):
        sentiment_scores.append(comprehensive_sentiment_analysis(input_dataframe.iloc[i][text_field], cat_dict, stem_dict, counts_dict))    
  
    sentiment_scores1 = np.asarray(sentiment_scores)
    sentiment_scores1 = np.squeeze(sentiment_scores1)
    if len(sentiment_scores1.shape) == 1:
        sentiment_scores1 = np.expand_dims(sentiment_scores1,axis=0)
    sentiment_scores1 = pd.DataFrame(sentiment_scores1,columns=seq2)
    return sentiment_scores1  
  
########## Do a sentiment analysis ##########
def comprehensive_sentiment_analysis(text, cat_dict, stem_dict, counts_dict):      
    if len(text) != 0:
        quotes, Exclaim, AllPunc, allcaps = Functions.stuff_LIWC_leftout(text)
        lex_div = Functions.ttr(text)
        counts_norm = Functions.POS_features(text)
        counts_norm = [str(c) for c in counts_norm]
        counts_norm_liwc, liwc_cats = Functions.LIWC(text, cat_dict, stem_dict, counts_dict)
        counts_norm_liwc = [str(c) for c in counts_norm_liwc]
        vadneg, vadneu, vadpos = Functions.vadersent(text)
        fke, SMOG = Functions.readability(text)
        stop, wordlen, WC = Functions.wordlen_and_stop(text)
        NB_pobj, NB_psubj = Functions.subjectivity(text)
        bias_count, assertives_count, factives_count, hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count = Functions.bias_lexicon_feats(text)
        HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral = Functions.moral_foundation_feats(text)
        happiness = Functions.happiness_index_feats(text)
#         dummy = [(happiness, HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral, bias_count, assertives_count, factives_count, hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count, lex_div,vadneg,vadneu,vadpos,fke,SMOG,stop,wordlen,WC,NB_pobj,NB_psubj,quotes,Exclaim,AllPunc,allcaps, counts_norm ,counts_norm_liwc)]
#         dummy = happiness, HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral, bias_count, assertives_count, factives_count, hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count, lex_div,vadneg,vadneu,vadpos,fke,SMOG,stop,wordlen,WC,NB_pobj,NB_psubj,quotes,Exclaim,AllPunc,allcaps, ",".join(counts_norm), ",".join(counts_norm_liwc)
        dummy = [happiness, HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral, bias_count, assertives_count, factives_count, hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count, lex_div,vadneg,vadneu,vadpos,fke,SMOG,stop,wordlen,WC,NB_pobj,NB_psubj,quotes,Exclaim,AllPunc,allcaps]
        counts_norm = np.asarray(counts_norm)
        counts_norm_liwc = np.asarray(counts_norm_liwc)
        dummy = np.asarray(dummy)
        output = np.concatenate((dummy, counts_norm, counts_norm_liwc), axis=0)
        output = np.expand_dims(output,axis=0)
        return output
    else:
        return 0
    

### Recommend new articles
def Recommend_new_articles(user_input_article_catg,article_fromWebSearch_info):
    #Remove articles from the same category as the user input's from the database
    article_fromWebSearch_info = article_fromWebSearch_info[article_fromWebSearch_info['bias'] != user_input_article_catg]
    if user_input_article_catg == 0:
        new_cats = [1,2]
        new_cats_subs = [3,4]
    elif user_input_article_catg == 1 or user_input_article_catg == 3:
        new_cats = [0,2]
        new_cats_subs = [0,4]
    elif user_input_article_catg == 2 or user_input_article_catg == 4:
        new_cats = [0,1]
        new_cats_subs = [0,3]
    sugg = []    
    for i in range(len(new_cats)):
        if not article_fromWebSearch_info[article_fromWebSearch_info.bias == new_cats[i]].empty:
            sugg.append(article_fromWebSearch_info[article_fromWebSearch_info.bias == new_cats[i]].iloc[0])
        elif not article_fromWebSearch_info[article_fromWebSearch_info.bias == new_cats_subs[i]].empty:
            sugg.append(article_fromWebSearch_info[article_fromWebSearch_info.bias == new_cats_subs[i]].iloc[0])
        else: 
            continue
    return sugg
        


#### Find new articels on the same topic as the user's input, process them and suggest new articles
def recommender_system(original_title, input_article_catg, start_google_search, end_google_search, loaded_model, one_hot_encoder):
##### Search for similar articles on Google news
    search_results_list = return_websearch_outputs(original_title, start_google_search, end_google_search)
    #Find more details on the outputs of the Google search
    article_fromWebSearch_info = []
    for search_result in search_results_list:
        dummy = search_each_source(search_result)
        if dummy:
            article_fromWebSearch_info.append(dummy)
    if not article_fromWebSearch_info:
        return []
    article_fromWebSearch_info = pd.DataFrame.from_dict(article_fromWebSearch_info)
    # Turn the article sources into binary variable and drop duplicate sources
    article_fromWebSearch_info['source'] = article_fromWebSearch_info.source.astype(str)
    # article_fromWebSearch_info.source, labels = pd.Categorical(pd.factorize(article_fromWebSearch_info.source)[0]), pd.Categorical(pd.factorize(article_fromWebSearch_info.source)[1])
    # Add urls
    search_result1 = pd.DataFrame(search_results_list)
    article_fromWebSearch_info['urls'] =search_result1
    # # Drop duplicate sources
    # article_fromWebSearch_info = article_fromWebSearch_info.drop_duplicates(subset='source', keep="first")
    #Text cleanup on each article
    call_data_cleaning_functions(article_fromWebSearch_info,'NewsTitle')
    call_data_cleaning_functions(article_fromWebSearch_info,'news_body')
    #Feature engineering
    sentiment_scores = call_comprehensive_sentiment_analysis(article_fromWebSearch_info,'NewsTitle') 
    one_hot_encoder_columns = functions.add_one_hot_encoder(article_fromWebSearch_info,'source', one_hot_encoder)   
    features_for_prediction = pd.concat([sentiment_scores,one_hot_encoder_columns],axis=1)
    
    SearchResults_article_catg = loaded_model.predict(features_for_prediction)         
    article_fromWebSearch_info['bias'] = SearchResults_article_catg
  
    ### Recommend new articles
    suggestions = Recommend_new_articles(input_article_catg,article_fromWebSearch_info)
    return suggestions
