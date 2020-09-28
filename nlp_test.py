from nltk.corpus import twitter_samples, stopwords #data
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist

import re, string

# positive_tweets = twitter_samples.strings('positive_tweets.json') #5000
# negative_tweets = twitter_samples.strings('negative_tweets.json') #5000
text = twitter_samples.strings('tweets.20150430-223406.json') #20000 - unknown sentiment
# tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
stop_words = stopwords.words('english')

#normalize words to infinitive/singular - run, running, ran
def lemmatize_token(token, tag, lemmatizer):
    if tag.startswith('NN'): #noun
        pos = 'n'
    elif tag.startswith('VB'): #verb
        pos = 'v'
    else:
        pos = 'a'
    return lemmatizer.lemmatize(token, pos)

#use regex to remove unnecessary characters and remove stopwords
def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    lemmatizer = WordNetLemmatizer()
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        token = lemmatize_token(token, tag, lemmatizer)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

all_pos_words = get_all_words(positive_cleaned_tokens_list)
all_neg_words = get_all_words(negative_cleaned_tokens_list)

freq_dist_pos = FreqDist(all_pos_words)
freq_dist_neg = FreqDist(all_neg_words)

print(freq_dist_pos.most_common(10))





