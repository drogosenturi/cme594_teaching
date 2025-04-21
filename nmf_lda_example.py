'''
code adapted from Sybil Derrible CME594 Homework 11
and https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
'''

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pdfplumber # library to read pdf into python
import nltk

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {0}: ".format(topic_idx))
        print (' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def extract_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

documents = []
documents.append(extract_text('minor-etal.pdf'))
documents.append(extract_text('garfinkel-etal.pdf'))

#Number of features to consider (i.e., individual token occurrence frequency)
no_features = 500

#Number of topics
components = 10
# check frequency of words
tokenize = nltk.tokenize.word_tokenize(documents[1])
doc_nltk = nltk.Text(tokenize)
doc_nltk.plot(40)

# ID stop words
stopwords = ["et","al.","the","a","is","'",".","(",")","in","of","and",
             "for","&",",","as","were","n","that", "al","to","their",
             "non","doi","no","https","we","with","an","whether","it",
             "or",":","by","are","be",";"]

''' NMF '''
# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(strip_accents="ascii",
                                   stop_words=stopwords)
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Run NMF
nmf = NMF(n_components=components, random_state=1, l1_ratio=.5).fit(tfidf)


''' LDA '''
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(strip_accents="ascii",
                                stop_words=stopwords)
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names_out()

# Run LDA
lda = LatentDirichletAllocation(n_components=components, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)


no_top_words = 5
print("RESULT FOR NMF\n")
display_topics(nmf, tfidf_feature_names, no_top_words)

print("\nRESULT FOR LDA\n")
display_topics(lda, tf_feature_names, no_top_words)