# -*- coding: utf-8 -*-
"""Streamlit_sub.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19JY1P9xLkValJCRORwo_bu2Wh-JJIc3c
"""

import pandas as pd
import numpy as np
import nltk
import random
nltk.download('punkt')
nltk.download('cmudict')
nltk.download('stopwords')
nltk.download('reuters')
nltk.download('words')
import spacy
import pysrt
import datetime
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict, stopwords, reuters, words as nltk_words
from nltk.probability import FreqDist
from textstat import textstat
import re
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,RobustScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from textstat import flesch_kincaid_grade, gunning_fog, smog_index
import torch
from sklearn.decomposition import PCA
import streamlit as st
nlp = spacy.load("en_core_web_sm")

language=['RUS','ENG']
selected_len = st.multiselect('Язык/Lsnguage ?', language)
if 'RUS' in selected_len:

  st.title('Проект: Определение уровня сложности субтитров на английском языке')

  st.write('''
    Цель проекта ,модель машинного обучения на на основании субтитров к фильмам или сериалам,
    показывает каким уровнем знания языка должен обладать человек что бы лучше понять эти субтитры,
    а так же получить короткую статистику о тексте.

  ''')

  st.write('')
  df=pd.read_csv('/content/drive/MyDrive/учеба/мастерская 2/subs.csv')





  class Features:
      d = cmudict.dict()
      word_freqs = FreqDist(i.lower() for i in reuters.words())
      common_words = set(nltk_words.words())
      stop_words = set(stopwords.words('english'))
      nlp = spacy.load('en_core_web_sm')
      
      def __init__(self, first):
          self.first = first
          self.clean_text = self.clean_html(self.first.text)
          self.sentences = sent_tokenize(self.clean_text)
          words = word_tokenize(self.clean_text)
          self.words = [word.lower() for word in words if word.isalpha()]
          self.non_stopwords = [word for word in self.words if word not in self.stop_words]
          self.complex_words = self.hard_words()
          self.doc = self.nlp(self.first.text)

      def clean_html(self, raw_html):
          cleanr = re.compile('<.*?>')
          intermediate_text = re.sub(cleanr, '', raw_html)
          cleantext = intermediate_text.replace('\n', ' ')
          return cleantext

      def avg_sent_length(self):
          return np.mean([len(sentence.split()) for sentence in self.sentences])

      def avg_word_length(self):
          return np.mean([len(word) for word in self.words])

      def difficult_words_ratio(self):
          return len(self.non_stopwords) / len(self.words) if self.words else 0

      def nsyl(self, word):
          return max([len(list(y for y in x if y[-1].isdigit())) for x in self.d.get(word.lower(), [])] or [0])

      def hard_words(self, frequency_threshold=5000):
          return {
              word for word in self.non_stopwords if len(word) > 2 and word in self.d and self.nsyl(word) > 2 and self.word_freqs[word] < frequency_threshold
          }

      def flesch_kincaid(self):
          total_sentences = len(self.sentences)
          total_words = len(self.words)
          total_syllables = sum([self.nsyl(word) for word in self.words])


          if total_words and total_sentences: 
              FK_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
              return FK_grade
          else:
              return 0

      def gunning_fog(self):
          total_sentences = len(self.sentences)
          total_words = len(self.words)
          complex_word_count = len(self.complex_words)
          
          
          if total_words and total_sentences:
              GF_index = 0.4 * ((total_words / total_sentences) + 100 * (complex_word_count / total_words))
              return GF_index
          else:
              return 0

      def smog_index(self):
          complex_word_count = len(self.complex_words)
          
          # SMOG index
          if complex_word_count:
              SMOG = 1.0430 * (30 * (complex_word_count))**.5 + 3.1291
              return SMOG
          else:
              return 0

      def lexical_diversity(self):
          num_subordinate_clauses = sum([1 for token in self.doc if token.dep_ in ('acl', 'advcl', 'csubj', 'ccomp', 'xcomp')])
          syntactic_complexity = num_subordinate_clauses / len(self.sentences)
          return syntactic_complexity

      def avg_time(self):
          seconds = []
          for i in self.first:
              start_time = datetime.datetime.combine(datetime.date.today(), i.start.to_time())
              end_time = datetime.datetime.combine(datetime.date.today(), i.end.to_time())
              seconds.append((end_time - start_time).total_seconds())
          return np.mean(seconds)




  df[[i for i in df.columns[2:-1]]]=df[[i for i in df.columns[2:-1]]].astype(float)

  df['Level']=np.where(df['Level']!='A2/A2+',df['Level'],'A2')
  df['Level']=np.where(df['Level']!='A2/A2+, B1',df['Level'],'B1')
  df['Level']=np.where(df['Level']!='B1, B2',df['Level'],'B2')

  df=df.loc[df['lexical_diversity']<2]


  st.write('Загрузите сюда ваши субтитры')
  uploaded_file = st.file_uploader("Choose a file",type=['.srt'])

  predict_data=pd.DataFrame()
  predict_data[[['subs','avg_sent_length','avg_word_length','lexical_diversity','flesch_kincaid','gunning','smog','time','difficult_words']]]=None

  module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
  model = hub.load(module_url)



  def vectorizer(data):

    subtitles = data.split('\n\n')

    
    pattern = re.compile('\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+.*\n|<.*?>|\[.*\]')

    
    
    subtitle_sentences = []
    for subtitle in subtitles:
          
          subtitle = re.sub(pattern, '', subtitle)

          
          subtitle_lines = subtitle.split('\n')

          
          for line in subtitle_lines:
              subtitle_sentences.extend(nltk.sent_tokenize(line))
    subtitle_sentences = [s for s in subtitle_sentences if re.match(r'[A-Z]', s)]
        
    subtitle_embeddings = model(subtitle_sentences)

      
    summary_vector = np.mean(subtitle_embeddings, axis=0)

    return summary_vector






  def column(data):
    return np.array(data).reshape(-1,1)








  X_vecorizer=np.array([vectorizer(i) for i in df['subs']])

  avg_sent_length=column(df['avg_sent_length'])
  avg_word_lengt=column(df['avg_word_length'])
  lexical_diversity=column(df['lexical_diversity'])
  flesch_kincaid=column(df['flesch_kincaid'])
  gunning=column(df['gunning'])
  smog=column(df['smog'])
  time=column(df['time'])
  difficult_words=column(df['difficult_words'])



  scaller=StandardScaler()
  X_other=np.concatenate([avg_sent_length,avg_word_lengt,lexical_diversity,flesch_kincaid,gunning,smog,time,difficult_words],axis=1)
  X_other_scaller=scaller.fit_transform(X_other)
  X_concate=np.concatenate([X_other_scaller,X_vecorizer],axis=1)


  features=X_concate
  label=LabelEncoder()
  target=label.fit_transform(df['Level'])

  features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.1,stratify=target,shuffle=True, random_state=122435)

  grid = CatBoostClassifier(
      iterations=1000,  
      learning_rate=0.1,  
      depth=6, 
      l2_leaf_reg=3, 
      loss_function='MultiClass', 
      random_seed=42 ,verbose=False)
      

  grid.fit(features_train,target_train)


  def preparing(data,subs,preprocesing):

    scaller=preprocesing
    X_vecorizer=np.array([vectorizer(subs)])
    avg_sent_length=column(data['avg_sent_length'])
    avg_word_lengt=column(data['avg_word_length'])
    lexical_diversity=column(data['lexical_diversity'])
    flesch_kincaid=column(data['flesch_kincaid'])
    gunning=column(data['gunning'])
    smog=column(data['smog'])
    time=column(data['time'])
    difficult_words=column(data['difficult_words'])

    X_other=np.concatenate([avg_sent_length,avg_word_lengt,lexical_diversity,flesch_kincaid,gunning,smog,time,difficult_words],axis=1)
    X_other_scaller=scaller.fit_transform(X_other)
    X_concate=np.concatenate([X_other_scaller,X_vecorizer],axis=1)

    return X_concate
    

  def highlight_text(text, color, font_size):
      highlighted_text = f'<span style="background-color: {color}; font-size: {font_size}px;">{text}</span>'
      return highlighted_text

  # Example usage
  highlighted = highlight_text('Информация о ваших субтитрах', 'yellow', 20)

  # Display the highlighted text
  st.markdown(highlighted, unsafe_allow_html=True)



  if uploaded_file is not None:

    file_content = uploaded_file.getvalue().decode()  # decode bytes to string
    subs = pysrt.from_string(file_content)
    x=Features(subs)
    predict_data.loc[0,['subs','avg_sent_length','avg_word_length','lexical_diversity','flesch_kincaid','gunning','smog','time','difficult_words']]=subs.text,x.avg_sent_length(),x.avg_word_length(),x.lexical_diversity(),x.flesch_kincaid(),x.gunning_fog(),x.smog_index(),x.avg_time(),x.difficult_words_ratio()


    text_from_subs=predict_data['subs'].to_string(index=False).replace('\n', ' ')
    predict=preparing(predict_data,text_from_subs,StandardScaler())



    
    
    sent_lenght=x.avg_sent_length()
    lexikal_deversity=x.lexical_diversity()
    flesh=x.flesch_kincaid()
    guning=x.gunning_fog()
    time=x.avg_time()
    hard=x.complex_words
    predict=label.inverse_transform(grid.predict(predict))
    result = highlight_text(f'Для понимания этого фильма вам необходимо обладать уровнем {predict[0]}', 'red', 20)
    st.markdown(result, unsafe_allow_html=True)
    answer=['Да','Нет']
    selected_models = st.multiselect('Подобрать фильмы с похожей категорией сложности ?', answer)
    if 'Да' in selected_models:
      number=st.number_input('Выберите число фильмов которые хотите получить,но не больше 4', min_value=1, max_value=4, value=2, step=1)
      data = [i for i in df.loc[df['Level']==predict[0]]['Movie']]
      random.shuffle(data)
      films=[]
      for i in range(number):
        films.append(data[random.randint(0,len(data))])
      st.write(f'Вот еще фильмы которые вам могут подойти по вашему уровню : {films}')

    st.write(f'Средняя длина предложения :{np.round(sent_lenght,2)}')
    st.write(f'Разнообразие слов в ваших субтитрах : {np.round(lexikal_deversity,2)} ')
    st.write(f'Среднее время отображения субтитров : {time}')
    st.write(f'Также вам могут встретиться сложные слова такие как {hard} ')

if 'ENG' in selected_len:
  st.title('Project: Determination of the level of difficulty of subtitles in English')

  st.write('''
  The goal of the project, a machine learning model based on subtitles for movies or TV series,
  shows what level of language knowledge a person should have in order to better understand these subtitles,
  as well as to get short statistics about the text.

  ''')

  st.write('')
  df=pd.read_csv('/content/drive/MyDrive/учеба/мастерская 2/subs.csv')
  nlp = spacy.load("en_core_web_sm")




  class Features:
      d = cmudict.dict()
      word_freqs = FreqDist(i.lower() for i in reuters.words())
      common_words = set(nltk_words.words())
      stop_words = set(stopwords.words('english'))
      nlp = spacy.load('en_core_web_sm')
      
      def __init__(self, first):
          self.first = first
          self.clean_text = self.clean_html(self.first.text)
          self.sentences = sent_tokenize(self.clean_text)
          words = word_tokenize(self.clean_text)
          self.words = [word.lower() for word in words if word.isalpha()]
          self.non_stopwords = [word for word in self.words if word not in self.stop_words]
          self.complex_words = self.hard_words()
          self.doc = self.nlp(self.first.text)

      def clean_html(self, raw_html):
          cleanr = re.compile('<.*?>')
          intermediate_text = re.sub(cleanr, '', raw_html)
          cleantext = intermediate_text.replace('\n', ' ')
          return cleantext

      def avg_sent_length(self):
          return np.mean([len(sentence.split()) for sentence in self.sentences])

      def avg_word_length(self):
          return np.mean([len(word) for word in self.words])

      def difficult_words_ratio(self):
          return len(self.non_stopwords) / len(self.words) if self.words else 0

      def nsyl(self, word):
          return max([len(list(y for y in x if y[-1].isdigit())) for x in self.d.get(word.lower(), [])] or [0])

      def hard_words(self, frequency_threshold=5000):
          return {
              word for word in self.non_stopwords if len(word) > 2 and word in self.d and self.nsyl(word) > 2 and self.word_freqs[word] < frequency_threshold
          }

      def flesch_kincaid(self):
          total_sentences = len(self.sentences)
          total_words = len(self.words)
          total_syllables = sum([self.nsyl(word) for word in self.words])


          if total_words and total_sentences: 
              FK_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
              return FK_grade
          else:
              return 0

      def gunning_fog(self):
          total_sentences = len(self.sentences)
          total_words = len(self.words)
          complex_word_count = len(self.complex_words)
          
          
          if total_words and total_sentences:
              GF_index = 0.4 * ((total_words / total_sentences) + 100 * (complex_word_count / total_words))
              return GF_index
          else:
              return 0

      def smog_index(self):
          complex_word_count = len(self.complex_words)
          
          # SMOG index
          if complex_word_count:
              SMOG = 1.0430 * (30 * (complex_word_count))**.5 + 3.1291
              return SMOG
          else:
              return 0

      def lexical_diversity(self):
          num_subordinate_clauses = sum([1 for token in self.doc if token.dep_ in ('acl', 'advcl', 'csubj', 'ccomp', 'xcomp')])
          syntactic_complexity = num_subordinate_clauses / len(self.sentences)
          return syntactic_complexity

      def avg_time(self):
          seconds = []
          for i in self.first:
              start_time = datetime.datetime.combine(datetime.date.today(), i.start.to_time())
              end_time = datetime.datetime.combine(datetime.date.today(), i.end.to_time())
              seconds.append((end_time - start_time).total_seconds())
          return np.mean(seconds)

  df[[i for i in df.columns[2:-1]]]=df[[i for i in df.columns[2:-1]]].astype(float)

  df['Level']=np.where(df['Level']!='A2/A2+',df['Level'],'A2')
  df['Level']=np.where(df['Level']!='A2/A2+, B1',df['Level'],'B1')
  df['Level']=np.where(df['Level']!='B1, B2',df['Level'],'B2')

  df=df.loc[df['lexical_diversity']<2]


  st.write('Upload your subtitles here')
  uploaded_file = st.file_uploader("Choose a file",type=['.srt'])

  predict_data=pd.DataFrame()
  predict_data[[['subs','avg_sent_length','avg_word_length','lexical_diversity','flesch_kincaid','gunning','smog','time','difficult_words']]]=None

  module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
  model = hub.load(module_url)



  def vectorizer(data):

    subtitles = data.split('\n\n')

    
    pattern = re.compile('\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+.*\n|<.*?>|\[.*\]')

    
    
    subtitle_sentences = []
    for subtitle in subtitles:
          
          subtitle = re.sub(pattern, '', subtitle)

          
          subtitle_lines = subtitle.split('\n')

          
          for line in subtitle_lines:
              subtitle_sentences.extend(nltk.sent_tokenize(line))
    subtitle_sentences = [s for s in subtitle_sentences if re.match(r'[A-Z]', s)]
        
    subtitle_embeddings = model(subtitle_sentences)

      
    summary_vector = np.mean(subtitle_embeddings, axis=0)

    return summary_vector






  def column(data):
    return np.array(data).reshape(-1,1)








  X_vecorizer=np.array([vectorizer(i) for i in df['subs']])

  avg_sent_length=column(df['avg_sent_length'])
  avg_word_lengt=column(df['avg_word_length'])
  lexical_diversity=column(df['lexical_diversity'])
  flesch_kincaid=column(df['flesch_kincaid'])
  gunning=column(df['gunning'])
  smog=column(df['smog'])
  time=column(df['time'])
  difficult_words=column(df['difficult_words'])



  scaller=StandardScaler()
  X_other=np.concatenate([avg_sent_length,avg_word_lengt,lexical_diversity,flesch_kincaid,gunning,smog,time,difficult_words],axis=1)
  X_other_scaller=scaller.fit_transform(X_other)
  X_concate=np.concatenate([X_other_scaller,X_vecorizer],axis=1)


  features=X_concate
  label=LabelEncoder()
  target=label.fit_transform(df['Level'])

  features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.1,stratify=target,shuffle=True, random_state=122435)

  grid = CatBoostClassifier(
      iterations=1000,  
      learning_rate=0.1,  
      depth=6, 
      l2_leaf_reg=3, 
      loss_function='MultiClass', 
      random_seed=42 ,verbose=False)
      

  grid.fit(features_train,target_train)


  def preparing(data,subs,preprocesing):

    scaller=preprocesing
    X_vecorizer=np.array([vectorizer(subs)])
    avg_sent_length=column(data['avg_sent_length'])
    avg_word_lengt=column(data['avg_word_length'])
    lexical_diversity=column(data['lexical_diversity'])
    flesch_kincaid=column(data['flesch_kincaid'])
    gunning=column(data['gunning'])
    smog=column(data['smog'])
    time=column(data['time'])
    difficult_words=column(data['difficult_words'])

    X_other=np.concatenate([avg_sent_length,avg_word_lengt,lexical_diversity,flesch_kincaid,gunning,smog,time,difficult_words],axis=1)
    X_other_scaller=scaller.fit_transform(X_other)
    X_concate=np.concatenate([X_other_scaller,X_vecorizer],axis=1)

    return X_concate
    

  def highlight_text(text, color, font_size):
      highlighted_text = f'<span style="background-color: {color}; font-size: {font_size}px;">{text}</span>'
      return highlighted_text

  # Example usage
  highlighted = highlight_text('Information about your subtitles', 'yellow', 20)

  # Display the highlighted text
  st.markdown(highlighted, unsafe_allow_html=True)



  if uploaded_file is not None:

    file_content = uploaded_file.getvalue().decode()  # decode bytes to string
    subs = pysrt.from_string(file_content)
    x=Features(subs)
    predict_data.loc[0,['subs','avg_sent_length','avg_word_length','lexical_diversity','flesch_kincaid','gunning','smog','time','difficult_words']]=subs.text,x.avg_sent_length(),x.avg_word_length(),x.lexical_diversity(),x.flesch_kincaid(),x.gunning_fog(),x.smog_index(),x.avg_time(),x.difficult_words_ratio()


    text_from_subs=predict_data['subs'].to_string(index=False).replace('\n', ' ')
    predict=preparing(predict_data,text_from_subs,StandardScaler())



    
    
    sent_lenght=x.avg_sent_length()
    lexikal_deversity=x.lexical_diversity()
    flesh=x.flesch_kincaid()
    guning=x.gunning_fog()
    time=x.avg_time()
    hard=x.complex_words
    predict=label.inverse_transform(grid.predict(predict))
    result = highlight_text(f'To understand this movie, you need to be level {predict[0]}', 'red', 20)
    st.markdown(result, unsafe_allow_html=True)

    answer=['Yes','No']
    selected_models = st.multiselect('Pick up movies with a similar category of difficulty ?', answer)
    if 'Yes' in selected_models:
      number=st.number_input('Enter the number of movies you want to get, but no more 4', min_value=1, max_value=4, value=2, step=1)
      data = [i for i in df.loc[df['Level']==predict[0]]['Movie']]
      random.shuffle(data)
      films=[]
      for i in range(number):
        films.append(data[random.randint(0,len(data))])
      st.write(f'Here are more films that may suit you according to your level : {films}')

    st.write(f'Average sentence length :{np.round(sent_lenght,2)}')
    st.write(f'The variety of words in your subtitles : {np.round(lexikal_deversity,2)} ')
    st.write(f'Average subtitle display time: {time}')
    st.write(f'You may also encounter complex words such as {hard} ')