# Document Keyword Similarity using LDA Topic Modeling

```python
import pandas as pd
import numpy as np
import nltk
```


```python
data = pd.read_csv('data/job_posts.csv', usecols=['Title', 'Company', 
                                                  'JobDescription', 'JobRequirement', 'RequiredQual'])
data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Company</th>
      <th>JobDescription</th>
      <th>JobRequirement</th>
      <th>RequiredQual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chief Financial Officer</td>
      <td>AMERIA Investment Consulting Company</td>
      <td>AMERIA Investment Consulting Company is seekin...</td>
      <td>- Supervises financial management and administ...</td>
      <td>To perform this job successfully, an\r\nindivi...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Full-time Community Connections Intern (paid i...</td>
      <td>International Research &amp; Exchanges Board (IREX)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>- Bachelor's Degree; Master's is preferred;\r\...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Country Coordinator</td>
      <td>Caucasus Environmental NGO Network (CENN)</td>
      <td>Public outreach and strengthening of a growing...</td>
      <td>- Working with the Country Director to provide...</td>
      <td>- Degree in environmentally related field, or ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BCC Specialist</td>
      <td>Manoff Group</td>
      <td>The LEAD (Local Enhancement and Development fo...</td>
      <td>- Identify gaps in knowledge and overseeing in...</td>
      <td>- Advanced degree in public health, social sci...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Software Developer</td>
      <td>Yerevan Brandy Company</td>
      <td>NaN</td>
      <td>- Rendering technical assistance to Database M...</td>
      <td>- University degree; economical background is ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.isnull().sum()
```




    Title               28
    Company              7
    JobDescription    3892
    JobRequirement    2522
    RequiredQual       484
    dtype: int64




```python
data = data.dropna()
data['Title'].value_counts()[:10]
```




    Accountant                                216
    Medical Representative                    151
    Chief Accountant                          151
    Sales Manager                             126
    Administrative Assistant                  124
    Lawyer                                    115
    Project Manager                            94
    Software Developer                         79
    Web Developer                              74
    Receptionist/ Administrative Assistant     73
    Name: Title, dtype: int64




```python
data = data[data['Title'].str.match('Software Developer|Data Analyst|Software Engineer' + 
                                          '|Web Developer|Web Designer')]
data.reset_index(drop=True, inplace=True)
data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Company</th>
      <th>JobDescription</th>
      <th>JobRequirement</th>
      <th>RequiredQual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Software Developer</td>
      <td>Synergy International Systems, Inc./Armenia</td>
      <td>Synergy International Systems, Inc./Armenia se...</td>
      <td>Specific tasks and key responsibilities includ...</td>
      <td>- Degree in Computer Science, Information Tech...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Web Designer</td>
      <td>ACRA Credit Bureau</td>
      <td>ACRA Credit Bureau seeks to fill the position ...</td>
      <td>Translate into Armenian and Russian a web-site...</td>
      <td>The successful candidate will demonstrate the\...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Web Developer/ Programmer</td>
      <td>"Click" Web Design</td>
      <td>The Web Developer/ Programmer will develop int...</td>
      <td>The Web Developer/ Programmer will be responsi...</td>
      <td>- At least 2 years experience as a Web Develop...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Web Designer</td>
      <td>"Click" Web Design</td>
      <td>The Web Designer will build flash based websites.</td>
      <td>The Web Designer will be responsible for creat...</td>
      <td>- At least 2 years experience as a web designe...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Software Developer</td>
      <td>Synergy International Systems, Inc. - Armenia</td>
      <td>The responsibilities of this position are focu...</td>
      <td>Specific tasks and key responsibilities includ...</td>
      <td>- Degree in Computer Science, Information Tech...</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.Title.value_counts()[:5]
```




    Software Developer    79
    Web Developer         74
    Software Engineer     42
    Web Designer          34
    Data Analyst           5
    Name: Title, dtype: int64




```python
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def remove_punc(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize(text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_doc = [w for w in word_tokens if not w in stop_words]
    return filtered_doc
```


```python
test = data.assign(desc=pd.Series(data.apply(lambda x: x.JobDescription + x.JobRequirement + x.RequiredQual, axis=1)))
```


```python
document = ""
for i in test.desc:
    document += i
```


```python
from collections import Counter

document = remove_punc(document.lower())
keywordList = preprocess(document)
ctr = Counter(keywordList)
ctr.most_common(10)
```




    [('knowledge', 880),
     ('experience', 808),
     ('software', 679),
     ('web', 608),
     ('work', 543),
     ('development', 539),
     ('design', 491),
     ('good', 379),
     ('ability', 371),
     ('skills', 339)]




```python
!pip install -U gensim
```

    Collecting gensim
      Downloading https://files.pythonhosted.org/packages/0b/66/04faeedb98bfa5f241d0399d0102456886179cabac0355475f23a2978847/gensim-3.8.3-cp37-cp37m-win_amd64.whl (24.2MB)
    Collecting smart-open>=1.8.1 (from gensim)
      Downloading https://files.pythonhosted.org/packages/ea/54/01525817b6f31533d308968b814999f7e666b2234f39a55cbe5de7c1ff99/smart_open-4.1.2-py3-none-any.whl (111kB)
    Collecting Cython==0.29.14 (from gensim)
      Downloading https://files.pythonhosted.org/packages/1f/be/b14be5c3ad1ff73096b518be1538282f053ec34faaca60a8753d975d7e93/Cython-0.29.14-cp37-cp37m-win_amd64.whl (1.7MB)
    Requirement already satisfied, skipping upgrade: scipy>=0.18.1 in e:\anaconda3\lib\site-packages (from gensim) (1.1.0)
    Requirement already satisfied, skipping upgrade: six>=1.5.0 in e:\anaconda3\lib\site-packages (from gensim) (1.12.0)
    Requirement already satisfied, skipping upgrade: numpy>=1.11.3 in e:\anaconda3\lib\site-packages (from gensim) (1.15.4)
    Installing collected packages: smart-open, Cython, gensim
      Found existing installation: Cython 0.28.5
        Uninstalling Cython-0.28.5:
          Successfully uninstalled Cython-0.28.5
    Successfully installed Cython-0.29.14 gensim-3.8.3 smart-open-4.1.2
    


```python
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
```

    E:\Anaconda3\lib\site-packages\requests\__init__.py:91: RequestsDependencyWarning: urllib3 (1.26.2) or chardet (3.0.4) doesn't match a supported version!
      RequestsDependencyWarning)
    


```python
def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    return stemmer.stem(lemmatizer.lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
```


```python
dictionary = gensim.corpora.Dictionary(test['desc'].map(preprocess))
```


```python
i = 0
for k, v in dictionary.iteritems():
    if i < 10:
        print(k, v)
        i += 1
    else:
        break
```

    0 abil
    1 addit
    2 analyt
    3 armenia
    4 aspect
    5 assur
    6 attent
    7 bachelor
    8 busi
    9 candid
    


```python
len(dictionary)
```




    1619




```python
dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
```


```python
len(dictionary)
```




    388




```python
bow_corpus = [dictionary.doc2bow(doc) for doc in test['desc'].map(preprocess)]
```


```python
from gensim import corpora, models
from pprint import pprint

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
```


```python
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
```


```python
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
```

    Topic: 0 
    Words: 0.019*"degre" + 0.018*"cod" + 0.018*"person" + 0.017*"plus" + 0.016*"motiv" + 0.015*"orient" + 0.015*"server" + 0.014*"technic" + 0.014*"high" + 0.014*"understand"
    Topic: 1 
    Words: 0.023*"technic" + 0.023*"server" + 0.019*"relat" + 0.019*"product" + 0.016*"project" + 0.014*"problem" + 0.013*"familiar" + 0.013*"manag" + 0.012*"look" + 0.012*"databas"
    Topic: 2 
    Words: 0.017*"provid" + 0.016*"system" + 0.015*"write" + 0.015*"technic" + 0.015*"relat" + 0.014*"technolog" + 0.013*"code" + 0.012*"engin" + 0.011*"manag" + 0.011*"javascript"
    Topic: 3 
    Words: 0.028*"plus" + 0.022*"project" + 0.021*"framework" + 0.019*"data" + 0.018*"creat" + 0.014*"databas" + 0.014*"network" + 0.014*"document" + 0.013*"unix" + 0.013*"respons"
    Topic: 4 
    Words: 0.024*"plus" + 0.019*"degre" + 0.019*"respons" + 0.016*"russian" + 0.016*"relat" + 0.015*"websit" + 0.014*"technic" + 0.012*"technolog" + 0.012*"armenian" + 0.012*"problem"
    Topic: 5 
    Words: 0.016*"javascript" + 0.016*"excel" + 0.014*"project" + 0.013*"respons" + 0.012*"high" + 0.012*"technolog" + 0.012*"understand" + 0.011*"mysql" + 0.011*"page" + 0.010*"exist"
    Topic: 6 
    Words: 0.040*"test" + 0.018*"manag" + 0.017*"requir" + 0.017*"product" + 0.016*"technic" + 0.015*"exist" + 0.015*"technolog" + 0.015*"plus" + 0.013*"base" + 0.013*"provid"
    Topic: 7 
    Words: 0.020*"data" + 0.019*"relat" + 0.019*"product" + 0.018*"excel" + 0.016*"technic" + 0.015*"implement" + 0.015*"task" + 0.014*"algorithm" + 0.014*"strong" + 0.013*"write"
    Topic: 8 
    Words: 0.018*"technolog" + 0.016*"manag" + 0.015*"engin" + 0.015*"websit" + 0.015*"technic" + 0.014*"requir" + 0.013*"databas" + 0.012*"test" + 0.011*"product" + 0.011*"plus"
    Topic: 9 
    Words: 0.020*"plus" + 0.020*"write" + 0.019*"desir" + 0.019*"familiar" + 0.019*"technolog" + 0.017*"framework" + 0.014*"excel" + 0.014*"technic" + 0.013*"relat" + 0.013*"prefer"
    


```python
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
```

    Topic: 0 Word: 0.014*"content" + 0.011*"data" + 0.011*"websit" + 0.011*"sit" + 0.008*"creat" + 0.008*"engin" + 0.008*"compani" + 0.008*"specif" + 0.008*"updat" + 0.007*"quick"
    Topic: 1 Word: 0.019*"physic" + 0.013*"math" + 0.010*"algorithm" + 0.010*"layout" + 0.010*"prefer" + 0.009*"oblig" + 0.009*"militari" + 0.009*"previous" + 0.009*"implement" + 0.009*"plus"
    Topic: 2 Word: 0.012*"thing" + 0.010*"give" + 0.010*"mobil" + 0.009*"practic" + 0.009*"jqueri" + 0.009*"implement" + 0.009*"bank" + 0.008*"task" + 0.008*"solut" + 0.008*"modifi"
    Topic: 3 Word: 0.011*"cod" + 0.010*"familiar" + 0.009*"member" + 0.009*"test" + 0.008*"provid" + 0.008*"framework" + 0.008*"technic" + 0.008*"databas" + 0.008*"document" + 0.007*"object"
    Topic: 4 Word: 0.010*"financi" + 0.009*"player" + 0.009*"layout" + 0.008*"accept" + 0.008*"person" + 0.008*"project" + 0.008*"system" + 0.008*"masteri" + 0.008*"sourc" + 0.008*"understand"
    Topic: 5 Word: 0.022*"cycl" + 0.021*"librari" + 0.019*"algorithm" + 0.018*"data" + 0.018*"comprehens" + 0.017*"engin" + 0.017*"linux" + 0.016*"structur" + 0.016*"engag" + 0.015*"oral"
    Topic: 6 Word: 0.015*"apach" + 0.011*"militari" + 0.011*"oblig" + 0.010*"linux" + 0.009*"educ" + 0.009*"administr" + 0.008*"accord" + 0.008*"server" + 0.008*"unix" + 0.008*"mysql"
    Topic: 7 Word: 0.015*"backend" + 0.012*"posit" + 0.010*"perform" + 0.010*"autom" + 0.009*"compani" + 0.009*"main" + 0.009*"prefer" + 0.009*"profession" + 0.009*"control" + 0.009*"process"
    Topic: 8 Word: 0.011*"excel" + 0.011*"adob" + 0.010*"graphic" + 0.010*"websit" + 0.009*"site" + 0.009*"interfac" + 0.008*"photoshop" + 0.008*"armenian" + 0.007*"respons" + 0.007*"flash"
    Topic: 9 Word: 0.020*"long" + 0.018*"term" + 0.018*"engag" + 0.017*"differ" + 0.013*"project" + 0.011*"creat" + 0.010*"expert" + 0.010*"look" + 0.010*"particip" + 0.010*"xhtml"
    


```python
for index, score in sorted(lda_model[bow_corpus[20]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
```

    
    Score: 0.9901075959205627	 
    Topic: 0.020*"data" + 0.019*"relat" + 0.019*"product" + 0.018*"excel" + 0.016*"technic" + 0.015*"implement" + 0.015*"task" + 0.014*"algorithm" + 0.014*"strong" + 0.013*"write"
    


```python
unseen_document = r'''
Under general supervision, formulates design\r\nstrategies, and participates in the
strategic planning of web site goals\r\nand objectives.- Participates in the overall
design structuring of the web sites;\r\norganizes and maintains the sites.
\r\n- Develops and implements plans to obtain and maintain a high level of\r\nfunctionality,
usability, and design structure for the web sites. \r\n- Assesses new standards, technologies 
and trends, and formulates\r\nstrategies and plans for future enhancement of web sites.
\r\n- Develops, and coordinates the creation of comprehensive graphic\r\nlayouts and elements
for new sections and/or features on the sites.- Strong proficiency with HTM/HTML, Dreamweaver,
Flash Technology,\r\nPhotoshop, Java-Script, CSS;\r\n- Familiarity with web templates;\r\n-
Advanced knowledge and understanding of web-based graphic design and\r\nlayout; \r\n-
Web planning and organizing skills;\r\n- Ability to evaluate new and evolving website
technologies; \r\n- Knowledge of a comprehensive range of web programming software and\r\nauthoring
languages; \r\n- Knowledge and understanding of internet operations and functionality,\r\nand of
a wide range of internet programming and design tools. \r\n- Web design experience and portfolio;
\r\n- Creation of work using your own innovations and by following the\r\nguidance of managers and
colleagues; \r\n- Self-organized and detailed oriented;\r\n- Strong inter-personal and communication
skills;\r\n- Efficient when under pressure;\r\n- Able to work independently;\r\n- Able to multi-task,
and adapt to flexible timelines.
'''
op = []
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    op.append("Score: {} Topic: {}".format(score, lda_model.print_topic(index, 5)))
```


```python
op
```




    ['Score: 0.3178059756755829 Topic: 0.016*"javascript" + 0.016*"excel" + 0.014*"project" + 0.013*"respons" + 0.012*"high"',
     'Score: 0.27502545714378357 Topic: 0.018*"technolog" + 0.016*"manag" + 0.015*"engin" + 0.015*"websit" + 0.015*"technic"',
     'Score: 0.2566033601760864 Topic: 0.020*"plus" + 0.020*"write" + 0.019*"desir" + 0.019*"familiar" + 0.019*"technolog"',
     'Score: 0.1427709013223648 Topic: 0.019*"degre" + 0.018*"cod" + 0.018*"person" + 0.017*"plus" + 0.016*"motiv"']




```python
lda_model.save('lda.model')
```

## Introduction
The system uses a pre-trained gensim LDA model on [Kaggle](https://www.kaggle.com/madhab/jobposts "this") dataset. It uses a Python/Flask backend to interact with it's web interface. The demo is available [Demo Link](https://resumeanalyzer.herokuapp.com "here").

## Overview

In natural language processing, the latent Dirichlet allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. For example, if observations are words collected into documents, it posits that each document is a mixture of a small number of topics and that each word's presence is attributable to one of the document's topics. LDA is an example of a topic model and belongs to the machine learning toolbox and in wider sense to the artificial intelligence toolbox.

We first identify the keywords in the title using our pre-trained model, first for the job description and then the resume textboxes. This is then simply compared with a cosine similarity metric to throw out a number and it's matching keyword to determine how it relates to our model data.

## User Interface

![UI](UI/home.png "Homepage")
![UI](UI/analyze.png "Main Page")
![UI](UI/login.png "Log In Page")
![UI](UI/register.png "Register Page")
