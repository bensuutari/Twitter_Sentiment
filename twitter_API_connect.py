from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sentiment_classify as s

openfile=open('/home/ben/Dropbox/python/Twitter_Sentiment/twitter_API_info.txt','rb')
readdata=openfile.read().split('\n')
openfile.close()

#consumer key, consumer secret, access token, access secret.
ckey=readdata[0].split('=')[1]
csecret=readdata[1].split('=')[1]
atoken=readdata[2].split('=')[1]
asecret=readdata[3].split('=')[1]

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        
        tweet = all_data["text"]
       	sentiment_value,confidence=s.sentiment(tweet)

        print(tweet)
	print('......................')
	print('sentiment value='+str(sentiment_value)+' confidence='+str(confidence))
	print('----------------------')
        time.sleep(2)
        return True

    def on_error(self, status):
        print status

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Trump"])
