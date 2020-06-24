import warnings
warnings.filterwarnings('ignore')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import newspaper
from newspaper import Article
from googlesearch import search_news

#### import packages used in EDA ####
import re
import nltk
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import sklearn

### packages for sentiment analysis
from resources.feature_functions import Functions
Functions = Functions()

tokenizer=TweetTokenizer()
lem = WordNetLemmatizer()


class DataCleaning_Classes(object):
    def __init__(self, df, text_field):
        #Initialization of the data structure
        self.data = df
        #Methods to fill in the data structure
        self.standardize_text(df, text_field)
        self.call_clean_text(df, text_field)
        self.tokenizer_text(df, text_field)

    ########## Standardize the text data ##########
    def standardize_text(self, df, text_field):
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"\\n", " ")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"http\S+", "")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"[$(),!?@\-\"\:\'\`\_\’\”\“\—\/\\]", " ")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"[0-9]", " ")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"http", "")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"@\S+", "")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"@", "at")
        self.data[text_field] = self.data[text_field].astype(str).str.lower()
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"\...", " ")

    ########## Do text lemmatization  ##########
    def get_wordnet_pos(self, treebank_tag):
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

    def clean_text(self, data):
        """
        This function receives comments and returns clean word-list
        """
        #Split the sentences into words
        words = tokenizer.tokenize(data)
        tagged = nltk.pos_tag(words)
        tags_treebank_tag = [self.get_wordnet_pos(w[1]) for w in tagged]
        for i, word in enumerate(words):
            if tags_treebank_tag[i] is None:
                words[i] = lem.lemmatize(word)
            else:
                words[i] = lem.lemmatize(word, tags_treebank_tag[i])
        # Removing stopwords
        words = [w for w in words if not w in eng_stopwords]
        clean_sent = " ".join(words)
        # remove any non alphanum,digit character
        return(clean_sent)

    def call_clean_text(self, data, text_field):
        clean_corpus = data[text_field].apply(self.clean_text)
        self.data[text_field] = clean_corpus

    ########## Tokenizing the text data ##########
    def tokenizer_text(self, df, text_field):
        from nltk.tokenize import RegexpTokenizer
        tokenizer = RegexpTokenizer(r'\w+')
        new_field = text_field + "_tokenized"
        self.data[new_field] = self.data[text_field].apply(tokenizer.tokenize)
        all_words = [word for tokens in self.data[new_field] for word in tokens]
        sentence_lengths = [len(tokens) for tokens in self.data[new_field]]
        VOCAB = sorted(list(set(all_words)))
        print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
        print("Max sentence length is %s" % max(sentence_lengths))



class SentimentAnalysis_Classes(object):
    def __init__(self, df, text_field):
        #Initialization of the data structure
        self.data = df
        self.Output = {}
        #Methods to fill in the data structure
        self.call_comprehensive_sentiment_analysis(df, text_field)

     #############################################################################
    """
    Functions for a more comprehensive sentiment analysis
    """
    #############################################################################
    def call_comprehensive_sentiment_analysis(self, input_dataframe, text_field):
        cat_dict, stem_dict, counts_dict = Functions.load_LIWC_dictionaries()
        liwc_cats = [cat_dict[cat] for cat in cat_dict]
        pos_tags = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","WP$","WRB","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP"]
        seq = ["Happiness","HarmVirtue","HarmVice","FairnessVirtue","FairnessVice","IngroupVirtue","IngroupVice","AuthorityVirtue","AuthorityVice","PurityVirtue","PurityVice","MoralityGeneral","bias_count","assertives_count","factives_count","hedges_count","implicatives_count","report_verbs_count","positive_op_count","negative_op_count","wneg_count","wpos_count","wneu_count","sneg_count","spos_count","sneu_count","TTR","vad_neg","vad_neu","vad_pos","FKE","SMOG","stop","wordlen","WC","NB_pobj","NB_psubj","quotes","Exclaim","AllPunc","allcaps"]
        seq2 = seq + pos_tags + liwc_cats

        sentiment_scores = []
        for i in range(input_dataframe.shape[0]):
            sentiment_scores.append(self.comprehensive_sentiment_analysis(input_dataframe.iloc[i][text_field], cat_dict, stem_dict, counts_dict))

        sentiment_scores1 = np.asarray(sentiment_scores)
        sentiment_scores1 = np.squeeze(sentiment_scores1)
        if len(sentiment_scores1.shape) == 1:
            sentiment_scores1 = np.expand_dims(sentiment_scores1,axis=0)
        sentiment_scores1 = pd.DataFrame(sentiment_scores1,columns=seq2)
        self.Output['Comp_sentiment_analysis_scores'] = sentiment_scores1

    ########## Do a sentiment analysis ##########
    def comprehensive_sentiment_analysis(self, text, cat_dict, stem_dict, counts_dict):
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
            dummy = [happiness, HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral, bias_count, assertives_count, factives_count, hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count, lex_div,vadneg,vadneu,vadpos,fke,SMOG,stop,wordlen,WC,NB_pobj,NB_psubj,quotes,Exclaim,AllPunc,allcaps]
            counts_norm = np.asarray(counts_norm)
            counts_norm_liwc = np.asarray(counts_norm_liwc)
            dummy = np.asarray(dummy)
            output = np.concatenate((dummy, counts_norm, counts_norm_liwc), axis=0)
            output = np.expand_dims(output,axis=0)
            return output
        else:
            return 0




#############################################################################
"""
Functions for pulling article info and do Google search
"""
#############################################################################
class Article_info_extraction(object):
    def __init__(self, search_term, input_URL, return_websearch_outputs, call_search_each_source):
        #Initialization of the data structure
        self.Output = {}
        #Methods to fill in the data structure
        self.return_websearch_outputs(return_websearch_outputs, search_term)
        self.search_each_source(input_URL, call_search_each_source)

    def return_websearch_outputs(self, return_websearch_outputs, search_term):
        if return_websearch_outputs:
            query = search_term
            search_results_list = []
            for result in search_news(str(query),        # The query you want to run
                            lang = 'en',  # The language
                            num = 10,     # Number of results per page
                            start = 0,    # First result to retrieve
                            stop = 10,  # Last result to retrieve
                            pause = 0.0001,  # Lapse between HTTP requests
                           ):
                search_results_list.append(result)
            self.Output['search_results_list'] = search_results_list
        else:
            pass

    def search_each_source(self, input_URL, call_search_each_source):
        if call_search_each_source:
            input_dataframe = dict();
            article = Article(input_URL)
            article.download();
            try:
                article.parse();
                input_dataframe['news_body'] = article.text
                input_dataframe['NewsTitle'] = article.title
                source = re.split("[.]",input_URL)[1]
                input_dataframe['source'] = source
                self.Output['input_dataframe'] = input_dataframe
            except newspaper.ArticleException:
                print("ArticleException error")
                pass
        else:
            pass



class Recommender_system(object):
    def __init__(self, user_input_article_catg, article_fromWebSearch_info):
        #Initialization of the data structure
        self.Output = {}
        #Methods to fill in the data structure
        self.Recommend_new_articles(user_input_article_catg, article_fromWebSearch_info)

    ### Recommend new articles
    def Recommend_new_articles(self, user_input_article_catg,article_fromWebSearch_info):
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
        self.Output['article_sugg'] = sugg
