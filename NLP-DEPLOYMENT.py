# Import Libraries *************************************************************
import pandas as pd
# import numpy as np
import re 
import nltk
nltk.download('punkt')

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from afinn import Afinn
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from nltk.util import ngrams
import re
from collections import Counter
import matplotlib.pyplot as plt
import string
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import coo_matrix
from spacy.lang.en import English


# Page Setup *************************************************************
st.set_page_config(layout="wide")
# front end elements of the web page
custom_title = "<h1 style='color: white; text-align: center;'>Hotel Review Analysis</h1>"

#creating an home button to redirect to the home page
def home_button():
    st.sidebar.button('Home')

home_button()

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.postimg.cc/v8KkstRD/hotel-image.jpg");
             background-attachment: fixed;
	         background-position: center;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()


custom_css = """
<style>
.logo{
position: absolute;
top: -25px;
right: -15px;
width: 100px;
height: auto;
z-index: 100px;
}
<style>
"""

# Your logo image tag without alt attribute
logo_html = '<img src="https://i.postimg.cc/1X6mNRh0/Excel-R-Logo.jpg" class="logo">'

# Combine title and logo in a single st.markdown() call
st.markdown(custom_css, unsafe_allow_html=True)  # Apply custom CSS
st.markdown(custom_title + logo_html, unsafe_allow_html=True)


# Function to add project members button
def project_members_button():
    if st.sidebar.button('Project Members'):
        with st.expander("Meet the Team"):
            st.write("- <span style='color: white;'>Gugloth Vijay</span>", unsafe_allow_html=True)
            st.write("- <span style='color: white;'>Manil</span>", unsafe_allow_html=True)
            st.write("- <span style='color: white;'>Pooja</span>", unsafe_allow_html=True)
            st.write("- <span style='color: white;'>Saloni</span>", unsafe_allow_html=True)

# Add an empty space or separator with white font color
text_color = "white"
st.write("", f'<p style="color: {text_color}; font-weight: bold;"></p>', unsafe_allow_html=True)

project_members_button()


# Data Cleaning *************************************************************
#Lemmatization
wordnet=WordNetLemmatizer()

#Stop word
stop_words=stopwords.words('english')

nlp=spacy.load("en_core_web_sm")

# Varibale created for words which are not included in the stopwords
not_stopwords = ("aren", "aren't", "couldn", "couldn't", "didn", "didn't",
                 "doesn", "doesn't", "don", "don't", "hadn", "hadn't", "hasn",
                 "hasn't", "haven", "haven't", "isn", "isn't", "mustn",
                 "mustn't", "no", "not", "only", "shouldn", "shouldn't",
                 "should've", "wasn", "wasn't", "weren", "weren't", "will",
                 "wouldn", "wouldn't", "won't", "very")
stop_words_ = [words for words in stop_words if words not in not_stopwords]

# Additional words added in the stop word list
stop_words_.append("I")
stop_words_.append("the")
stop_words_.append("s")

# Stop word for keyword extraction
stop_words_keywords = stopwords.words('english')

# special additioanl stop words added for keyword extraction
stop_words_keywords.extend([
    "will", "always", "go", "one", "very", "good", "only", "mr", "lot", "two",
    "th", "etc", "don", "due", "didn", "since", "nt", "ms", "ok", "almost",
    "put", "pm", "hyatt", "grand", "till", "add", "let", "hotel", "able",
    "per", "st", "couldn", "yet", "par", "hi", "well", "would", "I", "the",
    "s", "also", "great", "get", "like", "take", "thank"
])

#Pre-processing the new dataset
def processing(corpus):
    output=[]
    
    #convert to string
    review =str(corpus)
    
    #to handle punctuations
    review = re.sub('[^a-zA-Z0-9*]', ' ', review)
    
     # Converting Text to Lower case
    review = review.lower()

    # Spliting each words - eg ['I','was','happy']
    review = review.split()

    # Applying Lemmitization for the words eg: Argument -> Argue - Using Spacy Library
    review = nlp(' '.join(review))
    review = [token.lemma_ for token in review]

    # Removal of stop words
    review = [word for word in review if word not in stop_words_]

    # Joining the words in sentences
    review = ' '.join(review)
    output.append(review)
    
    return output


# Important Attributes *************************************************************
def keywords(corpus):
    output2=[]
    
    #convert to string
    review =str(corpus)
    
    #to handle punctuations
    review = re.sub('[^a-zA-Z0-9*]', ' ', review)
    
     # Converting Text to Lower case
    review = review.lower()

    # Spliting each words - eg ['I','was','happy']
    review = review.split()

    # Applying Lemmitization for the words eg: Argument -> Argue - Using Spacy Library
    review = nlp(' '.join(review))
    review = [token.lemma_ for token in review]

    # Removal of stop words
    review = [word for word in review if word not in stop_words_keywords]

    # Joining the words in sentences
    review = ' '.join(review)
    output2.append(review)
    
    tfidf2 = TfidfVectorizer(norm="l2",analyzer='word', stop_words=stop_words_keywords,ngram_range=(1,2))
    tfidf2_x = tfidf2.fit_transform(output2)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(tfidf2_x)
    
    # get feature names
    feature_names = tfidf2.get_feature_names_out()
    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(tfidf2.transform(output2))
    
    def sort_coo(coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    #sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    
    #extract only the top n, n here is 10
    def extract_topn_from_vector(feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""
        
        #use only topn items from vector
        sorted_items = sorted_items[:topn]
        
        score_vals = []
        feature_vals = []
        
        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            #keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
            
        #create a tuples of feature,score
        #results = zip(feature_vals,score_vals)
        results= feature_vals

        return pd.Series(results)
        
    attributes = extract_topn_from_vector(feature_names,sorted_items,10)
    
    return attributes


# Prediction *************************************************************
# following lines create boxes in which user can enter data required to make prediction
# Textbox for text user is entering

st.subheader("Enter the text you'd like to analyze.")
text = st.text_input('Enter text')  # text is stored in this variable

# when 'Button' is clicked, make the prediction and store it
if st.button("Predict"):
    cleaned = processing(text)

    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Analyze the sentiment of the review text
    sentiment = analyzer.polarity_scores(text)

    # Determine the sentiment label
    if sentiment['compound'] >= 0.05:
        overall_sentiment = 'Positive'
        sentiment_text_color = "green"
        Reaction = '😄'
    elif sentiment['compound'] <= -0.05:
        overall_sentiment = 'Negative'
        sentiment_text_color = "red"
        Reaction = '😡'
    else:
        overall_sentiment = 'Neutral'
        sentiment_text_color = "orange"
        Reaction = '😐'

    # Display the results
    st.markdown(
        f'<p style="color: {sentiment_text_color}; font-weight: bold;">The Overall Sentiment of the review is {overall_sentiment} {Reaction}</p>',
        unsafe_allow_html=True
    )

    # Display sentiment percentages
    positive_percentage = (sentiment['pos'] * 100)
    negative_percentage = (sentiment['neg'] * 100)
    neutral_percentage = 100 - positive_percentage - negative_percentage
    st.write(f'<span style="color: white;">Percentage of Neutral Sentiment: {neutral_percentage:.2f}%</span>', unsafe_allow_html=True)
    st.write(f'<span style="color: white;">Percentage of Positive Sentiment: {positive_percentage:.2f}%</span>', unsafe_allow_html=True)
    st.write(f'<span style="color: white;">Percentage of Negative Sentiment: {negative_percentage:.2f}%</span>', unsafe_allow_html=True)


# Add a separator or space between the sentiment and important attributes
st.write("")

# Check if the "IMP Attributes" button is clicked
if st.button("IMP Attributes"):
    st.subheader("Important Attributes in Reviews")
    imp_att = keywords(text)
    
    # Apply custom CSS to change the text color for important attributes
    attr_text_color = "white"  # Change to the color you prefer for important attributes
    
    for i in imp_att:
        st.markdown(
            f'<p style="color: {attr_text_color}; font-weight: bold;">{i}</p>',
            unsafe_allow_html=True
        )


# Sidebar button for visualizing Word Cloud
if st.sidebar.button("Word Cloud"):
    st.sidebar.header("Word Cloud Visualization")

    # Generate a word cloud from the input text
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


# Sidebar button for visualizing N-gram Chart
if st.sidebar.button("N-gram"):
    st.sidebar.header("N-gram Visualization")
    n = st.sidebar.number_input("Select N for N-grams", min_value=1, max_value=5, step=1, value=2)

    # Tokenize the text and create N-grams
    tokens = nltk.word_tokenize(text)
    n_grams = list(nltk.ngrams(tokens, n))

    # Function to filter out non-alphanumeric N-grams
    def is_valid_ngram(ngram):
        return all(word.isalnum() or word in string.punctuation for word in ngram)

    # Filter out non-alphanumeric N-grams and count their frequency
    filtered_n_grams = [ngram for ngram in n_grams if is_valid_ngram(ngram)]
    n_gram_freq = Counter(filtered_n_grams)

    # Create a DataFrame for N-gram data
    n_gram_df = pd.DataFrame(n_gram_freq.most_common(10), columns=['N-gram', 'Frequency'])

    # Convert N-gram tuples to strings for x-axis
    n_gram_df['N-gram'] = n_gram_df['N-gram'].apply(lambda x: ' '.join(x))

    # Create a bar chart to visualize the N-gram frequencies using Matplotlib
    st.subheader("N-gram Chart")
    plt.figure(figsize=(10, 5))
    plt.bar(n_gram_df['N-gram'], n_gram_df['Frequency'])
    plt.xlabel('N-gram')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    st.pyplot(plt)
