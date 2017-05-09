#we want to pickle a classifier because the aglo takes a long time and you don't want to redo it every time....in the case of NaiveBayes it's actually very fast so no need to pickle but this is for proof of principle for slower algos
print 'start imports'
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import FreqDist
import nltk
from nltk.tokenize import word_tokenize
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier#import NLTK's SciKitLearn Wrapper
from nltk.corpus import movie_reviews
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB#nonbinary distribution=NB
from sklearn.linear_model import LogisticRegression,SGDClassifier#SGD=schochastic gradient descent
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI#inherit from the nltk classifier class
from statistics import mode
import sys
from nltk.corpus import stopwords
import time
import string
from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()
tagset=None
#reload(sys)
#sys.setdefaultencoding('Cp1252')

allowed_word_types=['J','V']

lemmatizer=WordNetLemmatizer()
print 'finished with imports'
parse_data=False




class VoteClassifier(ClassifierI):#inherit from ClassifierI from nltk
	def __init__(self,*classifiers):#pass list of classifiers
		self._classifiers=classifiers
	def classify(self,features):
		votes=[]
		for c in self._classifiers:
			v=c.classify(features)
			votes.append(v)
		return mode(votes)
	def confidence(self,features):
		votes=[]
		for c in self._classifiers:
			v=c.classify(features)
			votes.append(str(v))
		choice_votes=votes.count(mode(votes))#how many votes does each get?
		conf=float(choice_votes)/len(votes)#confidence measure
		return conf
	
def find_features(document,word_feats):
	words=word_tokenize(document)#we use word_tokenize here instead of set() because now we just have a long string, not a list of words
	pos=nltk.tag._pos_tag(words,tagset,tagger)
	words_lemmatized=[]
	for w in pos:
                if w[1][0] in allowed_word_types:
                        words_lemmatized.append(lemmatizer.lemmatize(w[0].lower(),convert_pos_tag(w[1])))

	features={}
	for w in word_feats:
		features[w]=(w in words_lemmatized)

	return features
def convert_pos_tag(tag):    
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

#########Load the parsed data and featuresets#############################
openallwords=open('all_words.pickle','r')
all_words=pickle.load(openallwords)
openallwords.close()
all_words=FreqDist(all_words)
all_words=all_words.most_common(3000)
word_features=list()
for i in all_words:
        word_features.append(i[0])
###########################################################################



###########################################################################
###########################################################################
###########################################################################

print '-------------LOAD CLASSIFERS-------------'
basedir='/home/ben/Dropbox/python/Twitter_Sentiment/pickled_classifiers/'
loadNaiveBayes=open(basedir+'naivebayes.pickle','rb')
classifier=pickle.load(loadNaiveBayes)
loadNaiveBayes.close()

loadfile=open(basedir+'naivebayes.pickle','rb')
classifier=pickle.load(loadfile)
loadfile.close()

loadfile=open(basedir+'MNB.pickle','rb')
MNB_classifier=pickle.load(loadfile)
loadfile.close()

loadfile=open(basedir+'BNB.pickle','rb')
BNB_classifier=pickle.load(loadfile)
loadfile.close()

loadfile=open(basedir+'LR.pickle','rb')
LogisticRegression_classifier=pickle.load(loadfile)
loadfile.close()

loadfile=open(basedir+'SGD.pickle','rb')
SGD_classifier=pickle.load(loadfile)
loadfile.close()

loadfile=open(basedir+'SVC.pickle','rb')
SVC_classifier=pickle.load(loadfile)
loadfile.close()

loadfile=open(basedir+'LinearSVC.pickle','rb')
LinearSVC_classifier=pickle.load(loadfile)
loadfile.close()

loadfile=open(basedir+'NuSVC.pickle','rb')
NuSVC_classifier=pickle.load(loadfile)
loadfile.close()

voted_classifier=VoteClassifier(classifier,MNB_classifier,BNB_classifier,LogisticRegression_classifier,SGD_classifier,LinearSVC_classifier,NuSVC_classifier)
###########################################################################
###########################################################################
###########################################################################

def sentiment(text):
	feats=find_features(text,word_features)
	return voted_classifier.classify(feats),voted_classifier.confidence(feats)



