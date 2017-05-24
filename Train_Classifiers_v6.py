import nltk
from nltk.corpus import sentence_polarity,twitter_samples
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier#import NLTK's SciKitLearn Wrapper
from nltk.classify import ClassifierI#inherit from the nltk classifier class
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB, BernoulliNB#multinomial binomial, gaussian nonbinary, bernoulli nonbinary
from sklearn.linear_model import LogisticRegression,SGDClassifier#logistic regression, stochastic gradient descent
from sklearn.svm import SVC,LinearSVC,NuSVC#suppor vector classifier, nu support vector classifier
from statistics import mode
import random
import pickle
import time
import string
import os

tagger = PerceptronTagger()
tagset = None

#Define which parts of speech to be used for training, J=Adjective,V=Verb,N=Noun,R=Adverb
allowed_word_types=['J','V']#['J','V','N','R']

parse_data=False
train_data=True

lemmatizer=WordNetLemmatizer()

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

if parse_data:
        documents=[]
        all_words=[]
        #########Parse Twitter Positive/Negative Tweets from NLTK Corpus#########
        postweets=twitter_samples.strings('positive_tweets.json')
        negtweets=twitter_samples.strings('negative_tweets.json')
        #########################################################################
        #########Parse Sentence Polarity Data from NLTK Corpus#########
        #first 5333 sentences are negative, last 5333 are positive
        for i in list(sentence_polarity.sents())[0:5333]:
                negtweets.append(' '.join(i))
        for i in list(sentence_polarity.sents())[5333:]:
                postweets.append(' '.join(i))
        ###############################################################

        ########################################################
        print('START POSITIVE')
        counter=0
        timestartpos=time.time()
        for tweet in postweets:
                documents.append((tweet,'pos'))
                words=word_tokenize(tweet)
                pos=nltk.tag._pos_tag(words,tagset,tagger)
                for w in pos:
                        if (w[1][0] in allowed_word_types) and (w[0] not in string.punctuation):
                                all_words.append(lemmatizer.lemmatize(w[0].lower(),convert_pos_tag(w[1])))
                if counter%100==0:
                        print 'passed '+str(counter)+' positive tweets'
                counter+=1
        timeendpos=time.time()
        print('Total time to finish positive tweets:'+str(timeendpos-timestartpos)+' seconds')
        ########################################################

        ########################################################
        
        print('START NEGATIVE')
        counter=0
        timestartneg=time.time()
        for tweet in negtweets:
                documents.append((tweet,'neg'))
                words=word_tokenize(tweet)
                pos=nltk.tag._pos_tag(words,tagset,tagger)
                for w in pos:
                        if (w[1][0] in allowed_word_types) and (w[0] not in string.punctuation):
                                all_words.append(lemmatizer.lemmatize(w[0].lower(),convert_pos_tag(w[1])))
                if counter%100==0:
                        print('passed '+str(counter)+' negative tweets')
                counter+=1
        timeendneg=time.time()     
        print('Total time to finish negative tweets:'+str(timeendneg-timestartneg)+' seconds')
        ########################################################

        
        save_all_words=open('all_words.pickle','wb')
        pickle.dump(all_words,save_all_words)#now we've saved the parsed word list
        save_all_words.close()
        save_documents=open('all_documents.pickle','wb')
        pickle.dump(documents,save_documents)
        save_documents.close()
        all_words=FreqDist(all_words)
        all_words=all_words.most_common(3000)
        #word_features=all_words.most_common(3000)
        #word_features=list(all_words.keys())
        word_features=list()
        for i in all_words:
                word_features.append(i[0])
        print('start featuresets')
        featuresetsstart=time.time()
        featuresets=[(find_features(tweet,word_features),category) for (tweet,category) in documents]
        savefeaturesets=open('featuresets.pickle','wb')
        pickle.dump(featuresets,savefeaturesets)
        savefeaturesets.close()
        featuresetsend=time.time()
        print('end featuresets')
        print('time to run featuresets:'+str(featuresetsend-featuresetsstart)+' seconds')
        

else:
        all_words_load=open('all_words.pickle','rb')
        all_words=pickle.load(all_words_load)
        all_words_load.close()
        
        documents_load=open('all_documents.pickle','rb')
        documents=pickle.load(documents_load)
        documents_load.close()
        all_words=FreqDist(all_words)
        word_features=list(all_words.most_common(3000))

        tstartloadfeaturesets=time.time()
        print('start load featuresets')
        load_featuresets=open('featuresets.pickle','rb')
        featuresets=pickle.load(load_featuresets)
        load_featuresets.close
        tendloadfeaturesets=time.time()
        print('end load featuresets')
        print('time to load featuresets='+str(tendloadfeaturesets-tstartloadfeaturesets)+' seconds')


random.shuffle(featuresets)
#use first 18,000 features to train and remaining to test (~10% of data for training)
training_set=featuresets[:18000]
testing_set=featuresets[18000:]
print('start training data')
if train_data:
	#use naive bayes
	classifier=nltk.NaiveBayesClassifier.train(training_set)
	save_classifier=open(os.getcwd()+'/pickled_classifiers/naivebayes.pickle','wb')
	pickle.dump(classifier,save_classifier)
	save_classifier.close()
	print('Trained Naive Bayes')
	
	#use multinomial nonbinary
	MNB_classifier=SklearnClassifier(MultinomialNB())
	MNB_classifier.train(training_set)
	save_classifier=open(os.getcwd()+'/pickled_classifiers/MNB.pickle','wb')
	pickle.dump(MNB_classifier,save_classifier)
	save_classifier.close()
	print('Trained Multinomial Bayes')
	
	#use bernoulli nonbinary
	BNB_classifier=SklearnClassifier(BernoulliNB())
	BNB_classifier.train(training_set)
	save_classifier=open(os.getcwd()+'/pickled_classifiers/BNB.pickle','wb')
	pickle.dump(BNB_classifier,save_classifier)
	save_classifier.close()
	print('Trained Bernoulli NB')

	#use logitic regression
	LogisticRegression_classifier=SklearnClassifier(LogisticRegression())
	LogisticRegression_classifier.train(training_set)
	save_classifier=open(os.getcwd()+'/pickled_classifiers/LR.pickle','wb')
	pickle.dump(LogisticRegression_classifier,save_classifier)
	save_classifier.close()
	print('Trained Logistic Regression')

	#use stochastic gradient descent classifier
	SGD_classifier=SklearnClassifier(SGDClassifier())
	SGD_classifier.train(training_set)
	save_classifier=open(os.getcwd()+'/pickled_classifiers/SGD.pickle','wb')
	pickle.dump(SGD_classifier,save_classifier)
	save_classifier.close()
	print('Trained Stochastic GD')

	#use support vector classifier
	SVC_classifier=SklearnClassifier(SVC())
	SVC_classifier.train(training_set)
	save_classifier=open(os.getcwd()+'/pickled_classifiers/SVC.pickle','wb')
	pickle.dump(SVC_classifier,save_classifier)
	save_classifier.close()
	print('Trained SVC')


	#use linear support vector classifier
	LinearSVC_classifier=SklearnClassifier(LinearSVC())
	LinearSVC_classifier.train(training_set)
	save_classifier=open(os.getcwd()+'/pickled_classifiers/LinearSVC.pickle','wb')
	pickle.dump(LinearSVC_classifier,save_classifier)
	save_classifier.close()
	print('Trained Linear SVC')

	#use Nu support vector classifier
	NuSVC_classifier=SklearnClassifier(NuSVC())
	NuSVC_classifier.train(training_set)
	save_classifier=open(os.getcwd()+'/pickled_classifiers/NuSVC.pickle','wb')
	pickle.dump(NuSVC_classifier,save_classifier)
	save_classifier.close()
	print('Trained Nu SVC')
else:
	classifier_f=open(os.getcwd()+'/pickled_classifiers/naivebayes.pickle',"rb")
	classifier=pickle.load(classifier_f)
	classifier_f.close()

	classifier_f=open(os.getcwd()+'/pickled_classifiers/MNB.pickle',"rb")
	MNB_classifier=pickle.load(classifier_f)
	classifier_f.close()

	classifier_f=open(os.getcwd()+'/pickled_classifiers/BNB.pickle',"rb")
	BNB_classifier=pickle.load(classifier_f)
	classifier_f.close()

	classifier_f=open(os.getcwd()+'/pickled_classifiers/LR.pickle',"rb")
	LogisticRegression_classifier=pickle.load(classifier_f)
	classifier_f.close()

	classifier_f=open(os.getcwd()+'/pickled_classifiers/SGD.pickle',"rb")
	SGD_classifier=pickle.load(classifier_f)
	classifier_f.close()

	classifier_f=open(os.getcwd()+'/pickled_classifiers/SVC.pickle',"rb")
	SVC_classifier=pickle.load(classifier_f)
	classifier_f.close()

	classifier_f=open(os.getcwd()+'/pickled_classifiers/LinearSVC.pickle',"rb")
	LinearSVC_classifier=pickle.load(classifier_f)
	classifier_f.close()

	classifier_f=open(os.getcwd()+'/pickled_classifiers/NuSVC.pickle',"rb")
	NuSVC_classifier=pickle.load(classifier_f)
	classifier_f.close()




print('Naive Bayes accuracy:'+str(nltk.classify.accuracy(classifier,testing_set)))
print('MNB_classifier accuracy:'+str(nltk.classify.accuracy(MNB_classifier,testing_set)))
print('BNB_classifier accuracy:'+str(nltk.classify.accuracy(BNB_classifier,testing_set)))
print('LogisticRegression_classifier accuracy:'+str(nltk.classify.accuracy(LogisticRegression_classifier,testing_set)))
print('SGDClassifier_classifier accuracy:'+str(nltk.classify.accuracy(SGD_classifier,testing_set)))
print('SVC_classifier accuracy:'+str(nltk.classify.accuracy(SVC_classifier,testing_set)))
print('LinearSVC_classifier accuracy:'+str(nltk.classify.accuracy(LinearSVC_classifier,testing_set)))
print('NuSVC_classifier accuracy:'+str(nltk.classify.accuracy(NuSVC_classifier,testing_set)))

voted_classifier=VoteClassifier(classifier,MNB_classifier,BNB_classifier,LogisticRegression_classifier,SGD_classifier,LinearSVC_classifier,NuSVC_classifier)
print('Voted_classifier accuracy:'+str(nltk.classify.accuracy(voted_classifier,testing_set)))


