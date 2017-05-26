#      Realtime Sentiment Analysis on Twitter         #


This program searches a keyword in Twitter and returns current Tweets containing the keyword and the predicted sentiment (positive/negative).

You'll have to have Twitter API access and create an app to use this program: [https://apps.twitter.com/](https://apps.twitter.com/)

Once you've got your Twitter API information (ckey, csecret, atoken, asecret), enter all of it into twitter_API_info.txt

Twitter_Sentiment uitilizes NLTK to parse sentence structure and determines sentence polarity with NLTK's perceptron tagger trained on the NLTK Twitter Corpus and the Sentence Polarity Corpus: [http://www.nltk.org/howto/corpus.html](http://www.nltk.org/howto/corpus.html/).

If you'd like to retrain the perceptron on different data, use Train_Classifiers.py.

If you'd like to simply run the sentiment bot with the data I trained on run Twitter_API_connect.py

**Give a a try!**



