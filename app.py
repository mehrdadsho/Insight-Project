#############################################################################
"""
Loading necessary packages
"""
#############################################################################
import pickle
import pandas as pd
import streamlit as st
import webbrowser
import functions
import timeit
#############################################################################


#############################################################################
#Load the model used in classifying the text data
#############################################################################
saved_model_path = "/home/ubuntu/StreamlitApp/Models/"
filename = saved_model_path + 'randomforest_Big_sentiment_analysis.sav'
loaded_model = pickle.load(open(filename, 'rb'))
filename = saved_model_path + 'count_vectorizer.sav'
count_vectorizer = pickle.load(open(filename, 'rb'))
filename = saved_model_path + 'tfidf_vectorizer.sav'
tfidf_vectorizer = pickle.load(open(filename, 'rb'))
filename = saved_model_path + 'one_hot_encoder.sav'
one_hot_encoder = pickle.load(open(filename, 'rb'))

Class_labels = ['Center', 'Lean Right', 'Lean Left', 'Right', 'Left']
Class_labels_titles = ['Center', 'Right', 'Left', 'Right', 'Left']

#############################################################################


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

local_css("/home/ubuntu/StreamlitApp/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

user_input = []
# icon("search")
user_input = st.text_input("", "")
button_clicked = st.button("Search")

if user_input:
    start = timeit.default_timer()
    #Find more details on the user-input article
    article_info = []
    dummy = functions.search_each_source(user_input)
    if dummy:
        article_info.append(dummy) 
    else:
        st.write("Sorry! no info on article was found")
        raise
    original_title = article_info[0]["NewsTitle"]
    input_article_info = pd.DataFrame.from_dict(article_info)
    #Text cleanup on each article
    functions.call_data_cleaning_functions(input_article_info,'NewsTitle')
    functions.call_data_cleaning_functions(input_article_info,'news_body')
    #Feature engineering
    sentiment_scores = functions.call_comprehensive_sentiment_analysis(input_article_info,'NewsTitle')     
    one_hot_encoder_columns = functions.add_one_hot_encoder(input_article_info,'source', one_hot_encoder)   
    features_for_prediction = pd.concat([sentiment_scores,one_hot_encoder_columns],axis=1)
    input_article_catg = loaded_model.predict(features_for_prediction)      
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
        
    
    ##### Search for similar articles on Google news and recommend new articles 
    start_google_search = 0
    suggestions = []

    
    while len(suggestions) == 0 and timeit.default_timer()-start <= 20:    
        end_google_search = start_google_search + 5
        important_features = functions.get_key_words(count_vectorizer,tfidf_vectorizer, original_title)
        suggestions = functions.recommender_system(important_features, input_article_catg[0], start_google_search, end_google_search,
                                               loaded_model, one_hot_encoder)
        start_google_search += 6
        from bokeh.models.widgets import Div
        if len(suggestions) > 0:
            strr_sug1 = 'From the ' + Class_labels_titles[suggestions[0]['bias']]
            if st.button(strr_sug1):
                webbrowser.open(suggestions[0]['urls'])
        if len(suggestions) > 1:
            strr_sug1 = 'From the ' + Class_labels_titles[suggestions[1]['bias']]
            if st.button(strr_sug1):
                webbrowser.open(suggestions[1]['urls'])
                
    if len(suggestions) == 0:
        st.markdown("<h1 style='text-align: center; color: black;'>Sorry! No Matching Articles Was Found!</h1>", unsafe_allow_html=True)
    
        
    

    # def Recommend_new_articles(user_input_article_catg,SearchResults_article_catg):
    #     #Remove articles from the same category as the user input's from the database
    #     Class_labels = ['Center', 'Lean Right', 'Lean Left', 'Right', 'Left', 'Mixed']

    #     # article_fromWebSearch_info = article_fromWebSearch_info.drop('source'==user_input_article_catg)
    #     # if user_input_article_catg == 0:
    #     #     new_cats = [1,2]
    #     # elif f user_input_article_catg == 1:
    #     #     new_cats = [0,2]
    #     # elif f user_input_article_catg == 2:
    #     #     new_cats = [0,1]
        


    


