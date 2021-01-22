import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from collections import Counter

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import models, corpora

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import  ngrams, RegexpTokenizer
from nltk.corpus import stopwords

from flask import request
from flask_wtf import FlaskForm

from wtforms import StringField
from wtforms.validators import DataRequired


class Analyze():
    def __init__(self, resume, jobdesc):
        self.resume = resume
        self.jobdesc = jobdesc

    def lemmatize_stemming(self, text):
        stemmer = SnowballStemmer('english')
        lemmatizer = WordNetLemmatizer()
        return stemmer.stem(lemmatizer.lemmatize(text, pos='v'))

    def preprocess(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result

    def ldaAnalyze(self):
        dictionary = gensim.corpora.Dictionary([[w] for w in self.resume.splitlines()])
        dictionary.filter_extremes(no_below=7, no_above=0.5, keep_n=100000)
        lda_model = models.LdaModel.load('model/lda.model')
        bow_vector = dictionary.doc2bow(self.preprocess(self.jobdesc))
        output = []
        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
            output.append("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
        return output

    def metrics(self):
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r"\w+")
        filtered = tokenizer.tokenize(self.jobdesc)
        words = [word for word in filtered if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        ngram_counts = Counter(ngrams(words, 1))
        results = [ {'keyword': str(tup[0][0]), 'freq': str(tup[1])} for tup in ngram_counts.most_common(15) ]
        return results


