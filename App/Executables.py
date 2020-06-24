#############################################################################
"""
Loading necessary packages
"""
#############################################################################
from __future__ import absolute_import, print_function
import pickle
import pandas as pd
import streamlit as st
import webbrowser
import App_Classes
#############################################################################


#############################################################################
#Load the model used in classifying the text data
#############################################################################
saved_model_path = "/Users/mehrdadshokrabadi/Documents/Miscellaneous/DS/Insight/Project/Models/"
filename = saved_model_path + 'randomforest_Big_sentiment_analysis.sav'
loaded_model = pickle.load(open(filename, 'rb'))
Class_labels = ['Center', 'Lean Right', 'Lean Left', 'Right', 'Left']
Class_labels_titles = ['Center', 'Right', 'Left', 'Right', 'Left']



#############################################################################
#load the streamlit app and take input from the user
#############################################################################
st.markdown("<h1 style='text-align: center; color: black;'>BubbleBurst</h1>", unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("/Users/mehrdadshokrabadi/Documents/Miscellaneous/DS/Insight/StreamlitApp/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

user_input = []
# icon("search")
user_input = st.text_input("", "")
button_clicked = st.button("Search")

if user_input:
    #Find more details on the user-input article
    article_info = []
    dummy = App_Classes.Article_info_extraction(user_input, 'Nothing', return_websearch_outputs=False, call_search_each_source=True).Output['input_dataframe']
    if dummy:
        article_info.append(dummy)
    else:
        st.write("Sorry! no info on article was found")
        raise
    original_title = article_info[0]["NewsTitle"]
    input_article_info = pd.DataFrame.from_dict(article_info)
    #Text cleanup on each article
    input_article_info = App_Classes.DataCleaning_Classes(input_article_info,'NewsTitle').data
    #Feature engineering
    sentiment_scores = App_Classes.SentimentAnalysis_Classes(input_article_info,'NewsTitle')
    input_article_catg = loaded_model.predict(sentiment_scores)
    # Print out the output
    st.markdown("<h1 style='text-align: center; color: black;'>Your article's bias is</h1>", unsafe_allow_html=True)
    if Class_labels[input_article_catg[0]] == 'Right' or Class_labels[input_article_catg[0]] == 'Lean Right':
        strr = "<h1 style='text-align: center; color: red;'>" + str(Class_labels[input_article_catg[0]]) + "</h1>"
        st.markdown(strr, unsafe_allow_html=True)
    elif Class_labels[input_article_catg[0]] == 'Left' or Class_labels[input_article_catg[0]] == 'Lean Left':
        strr = "<h1 style='text-align: center; color: blue;'>" + str(Class_labels[input_article_catg[0]]) + "</h1>"
        st.markdown(strr, unsafe_allow_html=True)
    else:
        strr = "<h1 style='text-align: center; color: purple;'>" + str(Class_labels[input_article_catg[0]]) + "</h1>"
        st.markdown(strr, unsafe_allow_html=True)


    ##### Search for similar articles on Google news
    search_results_list = App_Classes.Article_info_extraction('Nothing', original_title, return_websearch_outputs=True, call_search_each_source=False).Output['input_dataframe']
    #Find more details on the outputs of the Google search
    article_fromWebSearch_info = []
    for search_result in search_results_list:
        dummy = dummy = App_Classes.Article_info_extraction('Nothing', search_result, return_websearch_outputs=False, call_search_each_source=True).Output['input_dataframe']
        if dummy:
            article_fromWebSearch_info.append(dummy)
    article_fromWebSearch_info = pd.DataFrame.from_dict(article_fromWebSearch_info)
    # Turn the article sources into binary variable and drop duplicate sources
    article_fromWebSearch_info['source'] = article_fromWebSearch_info.source.astype(str)
    article_fromWebSearch_info.source, labels = pd.Categorical(pd.factorize(article_fromWebSearch_info.source)[0]), pd.Categorical(pd.factorize(article_fromWebSearch_info.source)[1])
    # Add urls
    search_result1 = pd.DataFrame(search_results_list)
    article_fromWebSearch_info['urls'] =search_result1
    #Text cleanup on each article
    article_fromWebSearch_info = App_Classes.DataCleaning_Classes(article_fromWebSearch_info,'NewsTitle').data
    #Feature engineering
    sentiment_scores = App_Classes.SentimentAnalysis_Classes(article_fromWebSearch_info,'NewsTitle')
    SearchResults_article_catg = loaded_model.predict(sentiment_scores)
    article_fromWebSearch_info['bias'] = SearchResults_article_catg

    ### Recommend new articles
    suggestions = App_Classes.Recommender_system(input_article_catg[0],article_fromWebSearch_info).Output['article_sugg']

    from bokeh.models.widgets import Div
    if len(suggestions) > 0:
        strr_sug1 = 'From the ' + Class_labels_titles[suggestions[0]['bias']]
        if st.button(strr_sug1):
            webbrowser.open_new_tab(suggestions[0]['urls'])
    if len(suggestions) > 1:
        strr_sug1 = 'From the ' + Class_labels_titles[suggestions[1]['bias']]
        if st.button(strr_sug1):
            webbrowser.open_new_tab(suggestions[1]['urls'])
