import warnings
warnings.filterwarnings('ignore')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#### import packages used in EDA ####
import seaborn as sns
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
<<<<<<< HEAD
from resources.feature_functions import Functions
Functions = Functions()
=======
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48

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
<<<<<<< HEAD
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"\\n", " ")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"http\S+", "")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"[$(),!?@\-\"\:\'\`\_\’\”\“\—\/\\]", " ")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"[0-9]", " ")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"http", "")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"@\S+", "")
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"@", "at")
        self.data[text_field] = self.data[text_field].astype(str).str.lower()
        self.data[text_field] = self.data[text_field].astype(str).str.replace(r"\...", " ")
=======
        self.data[text_field] = self.data[text_field].str.replace(r"http\S+", "")
        self.data[text_field] = self.data[text_field].str.replace(r"http", "")
        self.data[text_field] = self.data[text_field].str.replace(r"@\S+", "")
        self.data[text_field] = self.data[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
        self.data[text_field] = self.data[text_field].str.replace(r"@", "at")
        self.data[text_field] = self.data[text_field].str.lower()
        self.data[text_field] = self.data[text_field].str.replace(r"\...", " ")
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48

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



class FeatureEngineering_Classes(object):
    def __init__(self, df, text_field, min_df, max_features, list_of_ngrams):
        #Initialization of the data structure
        self.data = df
        self.Output = {}
        #Methods to fill in the data structure
        self.create_bag_of_words(df, text_field, min_df)
        self.create_tfidf_features()
        self.feature_engineering(df, text_field)
        self.call_create_ngrams_features(df, text_field, min_df, max_features, list_of_ngrams)

    ########## Create vectorize bag of words ##########
    def create_bag_of_words(self, data, text_field, min_df):
        count_vectorizer = sklearn.feature_extraction.text.CountVectorizer()
        self.Output['count_vectorizer'] = sklearn.feature_extraction.text.CountVectorizer()
        self.Output['cv_emb'] = count_vectorizer.fit_transform(data[text_field])

    ########## Create term-frequency times inverse document-frequency (tf-idf) features ##########
    def create_tfidf_features(self):
        tfidf_vectorizer = sklearn.feature_extraction.text.TfidfTransformer()
        self.Output['Tfidf_emb'] = tfidf_vectorizer.fit_transform(self.Output['cv_emb'])
        self.Output['tfidf_vectorizer'] = tfidf_vectorizer

    def create_ngrams_features(self, data, text_column, min_df, max_features, ngram_range):
        tfv = sklearn.feature_extraction.text.TfidfVectorizer(max_features=max_features,
                                                        strip_accents='unicode', analyzer='word',ngram_range=ngram_range,
                                                        use_idf=1,smooth_idf=1,sublinear_tf=1)
        df = tfv.fit(data[text_column])
        features = np.array(tfv.get_feature_names())
        ngram_sparseMatrix =  tfv.transform(data[text_column])
        return tfv, features, ngram_sparseMatrix

    def call_create_ngrams_features(self, data, text_field, min_df, max_features, list_of_ngrams):
        print(len(list_of_ngrams))
        for i in range(len(list_of_ngrams)):
            new_field = text_field + "_" +str(list_of_ngrams[i][0]) +  "_grams"
            tfv, features, ngram_sparseMatrix = self.create_ngrams_features(data, text_field, min_df, max_features, list_of_ngrams[i])
            self.Output[new_field] = pd.DataFrame(ngram_sparseMatrix.toarray(), columns=features)

    ########## Engineer New Features ##########
    def feature_engineering(self, data, text_field):
        ## Indirect features
        eng_stopwords = set(stopwords.words("english"))
        #Sentense count in each comment:
        new_field = text_field + "_count_sent"
        self.Output[new_field]=self.data[text_field].apply(lambda x: len(re.findall(r"\w+\.+ ",str(x)))+1)
        #Word count in each comment:
        new_field = text_field + "_count_word"
        self.Output[new_field]=self.data[text_field].apply(lambda x: len(str(x).split()))
        #Unique word count
        new_field = text_field + "_count_unique_word"
        self.Output[new_field]=self.data[text_field].apply(lambda x: len(set(str(x).split())))
        #Letter count
        new_field = text_field + "_count_letters"
        self.Output[new_field]=self.data[text_field].apply(lambda x: len(str(x)))
        #punctuation count
        new_field = text_field + "_count_punctuations"
        self.Output[new_field] =self.data[text_field].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
        #title case words count
        new_field = text_field + "_count_words_title"
        self.Output[new_field] = self.data[text_field].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
        #Number of stopwords
        new_field = text_field + "_count_stopwords"
        self.Output[new_field] = self.data[text_field].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
        #Average length of the words
        new_field = text_field + "_mean_word_len"
        self.Output[new_field] = self.data[text_field].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


class SentimentAnalysis_Classes(object):
    def __init__(self, df, text_field):
        #Initialization of the data structure
        self.data = df
        self.Output = {}
        #Methods to fill in the data structure
<<<<<<< HEAD
        # self.call_sentiment_analyzer_scores_vader(df, text_field)
        # self.call_sentiment_analyzer_scores_TextBlob(df, text_field)
        self.call_comprehensive_sentiment_analysis(df, text_field)
=======
        self.call_sentiment_analyzer_scores_vader(df, text_field)
        self.call_sentiment_analyzer_scores_TextBlob(df, text_field)
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48

    ########## Do a sentiment analysis ##########
    def sentiment_analyzer_scores_vader(self, sentence):
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(sentence)
        #print("{:-<40} {}".format(sentence, str(score)))
        return score

    def sentiment_analyzer_scores_TextBlob(self, sentence):
        """
        The polarity score is a float within the range [-1.0, 1.0].
        The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
        """
        text = TextBlob(sentence)
        score = text.sentiment
        return score

    def sentiment_analyzer_scores_NRC(self, Data, emotions, emolex_words_set, emolex_words):
        '''
        Takes a DataFrame and a specified column of text and adds 10 columns to the
        DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
        column containing the value of the text in that emotions
        INPUT: DataFrame, string
        OUTPUT: the original DataFrame with ten new columns
        '''
        emo_df = np.zeros((len(emotions)))
        counter = 0
        for word in Data:
            if word in emolex_words_set:
                emo_score = emolex_words[emolex_words.word == word]
                counter += 1
                emo_df = emo_df + emo_score.iloc[0][1:]
        if counter != 0:
            emo_df = emo_df/counter
        return emo_df

    def call_sentiment_analyzer_scores_vader(self, data, text_field):
        VADER_scores = data[text_field].apply(self.sentiment_analyzer_scores_vader)
        VADER_scores_compund = [VADER_scores[i]['compound'] for i in range(len(VADER_scores))]
        new_field = text_field + "_VADER_scores_compund"
        self.Output[new_field] = VADER_scores_compund

    def call_sentiment_analyzer_scores_TextBlob(self, data, text_field):
        TextBlob_scores = data[text_field].apply(self.sentiment_analyzer_scores_TextBlob)
        TextBlob_scores_polarity = [TextBlob_scores[i][0] for i in range(len(TextBlob_scores))]
        new_field = text_field + "_TextBlob_scores_polarity"
        self.Output[new_field] = TextBlob_scores_polarity
        TextBlob_scores_subjectivity = [TextBlob_scores[i][1] for i in range(len(TextBlob_scores))]
        new_field = text_field + "_TextBlob_scores_subjectivity"
        self.Output[new_field] = TextBlob_scores_subjectivity

    def call_sentiment_analyzer_scores_NRC(self, data, text_field):
        filepath = ('/Users/mehrdadshokrabadi/Downloads/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
        emolex_df = pd.read_csv(filepath, names=["word", "emotion", "association"],sep='\t')
        emolex_words = emolex_df.pivot(index='word', columns='emotion',
                                           values='association').reset_index()
        emotions = emolex_words.columns.drop('word')
        emolex_words_set = set(emolex_words.word)
        NRCemo_scores_NewsTitle = []
        text_field_tokenized = text_field + "_tokenized"
        for i in range(data.shape[0]):
            NRCemo_scores_NewsTitle.append(self.sentiment_analyzer_scores_NRC(data.iloc[i][text_field_tokenized],
                                                                  emotions, emolex_words_set, emolex_words))
        NRCemo_scores_NewsTitle_pd = pd.DataFrame(NRCemo_scores_NewsTitle)
        new_field = text_field + "_NRC_scores_compund"
        self.Output[new_field] = NRCemo_scores_NewsTitle_pd

<<<<<<< HEAD
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


=======
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48
class word2vec_analysis(object):
    def __init__(self, word2vec, df, text_field):
        #Initialization of the data structure
        self.Output = {}
        #Methods to fill in the data structure
        self.call_get_word2vec_embeddings(word2vec, df, text_field)
    ########## run genism's word2vec on the dataset ##########
    def get_average_word2vec(self, tokens_list, vector, generate_missing=False, k=300):
        if len(tokens_list)<1:
            return np.zeros(k)
        if generate_missing:
            vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
        else:
            vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
        length = len(vectorized)
        summed = np.sum(vectorized, axis=0)
        averaged = np.divide(summed, length)
        return averaged
    def get_word2vec_embeddings(self, vectors, df, tokens_column, generate_missing=False):
        embeddings = df[tokens_column].apply(lambda x: self.get_average_word2vec(x, vectors,
                                                                                    generate_missing=generate_missing))
        return list(embeddings)
    def call_get_word2vec_embeddings(self, word2vec, data, text_field):
        new_field = text_field + "_word2vec"
        embeddings= self.get_word2vec_embeddings(word2vec, data, text_field)
        embeddings = pd.DataFrame(embeddings)
        self.Output[new_field] = embeddings



<<<<<<< HEAD

=======
>>>>>>> 0a45ca3eccb7e5bc2fd5ab99c4a01e62e61e4d48
