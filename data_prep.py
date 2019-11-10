import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import nltk
from sklearn.model_selection import GridSearchCV
nltk.download('wordnet')
from sklearn.svm import SVC 

def df_prep(df,cutoff=.5,margin=.05):
	
	helpful_percentage = []
	for pos, total in zip(df['8'],df['9']):
	    if total>0:
	        helpful_percentage.append(float(pos)/float(total))
	    else:
	        helpful_percentage.append(0)
	df['15'] = helpful_percentage
	margin_mask = [x>(cutoff+margin) or x<(cutoff-margin) for x in helpful_percentage]
	df = df[margin_mask]
	#create target variable base on helpfulness
	Corpus = pd.DataFrame()
	Corpus['text'] = df['13']
	lst = []
	for x in df['15']:
	    if x>=cutoff:
	        lst.append('good')
	    else:
	        lst.append('bad')
	#Create New DF with Cleaned Text
	Corpus['label'] = lst
	# Step - a : Remove blank rows if any.
	Corpus['text'].dropna(inplace=True)
	# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
	Corpus['text'] = [entry.lower() for entry in Corpus['text']]
	# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
	Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
	# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
	# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
	tag_map = defaultdict(lambda : wn.NOUN)
	tag_map['J'] = wn.ADJ
	tag_map['V'] = wn.VERB
	tag_map['R'] = wn.ADV
	lst = []
	for index,entry in enumerate(Corpus['text']):
	    
	    if index%1000 ==0:
	        print(index)
	    # Declaring Empty List to store the words that follow the rules for this step
	    Final_words = []
	    # Initializing WordNetLemmatizer()
	    word_Lemmatized = WordNetLemmatizer()
	    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
	    
	    for word, tag in pos_tag(entry):
	        # Below condition is to check for Stop words and consider only alphabets
	        if word not in stopwords.words('english') and word.isalpha():
	            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
	            Final_words.append(word_Final)
	    # The final processed set of words for each iteration will be stored in 'text_final'
	    
	    #Corpus.loc[index,'text_final'] = str(Final_words)
	    lst.append(str(Final_words))
	    
	Corpus['text_final'] = lst
	Corpus['help_score'] = df['15']
	Corpus['help_votes'] = df['9']
	Corpus['stars'] = df['7']
	return Corpus

def vectorize_df(Train_X, Test_X, Train_Y, Test_Y,method='TF_IDF'):
	Encoder = LabelEncoder()
	Train_Y = Encoder.fit_transform(Train_Y)
	Test_Y = Encoder.fit_transform(Test_Y)
	if method =='TF_IDF':
		Tfidf_vect = TfidfVectorizer(max_features=100000,ngram_range=(1, 5))
		Tfidf_vect.fit(Train_X)
		Train_X_Tfidf = Tfidf_vect.transform(Train_X)
		Test_X_Tfidf = Tfidf_vect.transform(Test_X)

		return Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y
	if method == 'W2V':
		print(1)




