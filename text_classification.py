import os
import pdb
import json
import numpy
import pandas
import argparse
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,multilabel_confusion_matrix

class opts(object):
  def __init__(self):
            self.parser = argparse.ArgumentParser()
            self.parser.add_argument('task', default='train', choices=['train','test'], help='specify the task as train or test')
            self.parser.add_argument('--epochs', default=5, type=int, help='default epochs are 5 | specify the training epochs')
            self.parser.add_argument('--batch_size', default=32, type=int, help='specify the batch size for training')
            self.parser.add_argument('--model_path', default='./model/maruti_voc', help='specify the model path to save and reuse the saved models')
            self.parser.add_argument('--train_data_path', default='social_media_tracker_correct.csv', help='specify the training data path')
            self.parser.add_argument('--test_data_path', default='social_media_tracker_correct.csv', help='specify the test data path')
            self.parser.add_argument('--classes', default=['Other','Feedback','Complaints'], nargs='+', help='specify the classes to be classified')
            self.parser.add_argument('--word_len', default=100, type=int, help='specify the maximum input text length')
            self.parser.add_argument('--vocab_size', default=10000, type=int, help='specify the unique vocabulary length of the input data')
            self.parser.add_argument('--truncation_type', default='post',type=str, help='specify the text truncation type to maintain the same text length for all the input samples for training the model')
            self.parser.add_argument('--padding_type', default='post', type=str, help='specify the padding type to maintain the same text length for all the input samples for training the model') #specify the padding options as well
            self.parser.add_argument('--confusion_matrix_save_path', default='confusion_matrix.jpg', help='specify the path to save the confusion matrix')
            self.parser.add_argument('--embeddings_path', default='glove.6B.100d.txt', help='specify the word embeddings path')
    
  def parse(self,args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)
    return opt

class TextClassification():
  def __init__(self,**options):
    self.options = options
    self.model_path = options.get('model_path','./model/maruti_voc')
    self.ready_to_test = self.load_config(self.model_path)
    if self.ready_to_test:
      pass
    else:
      self.vocab_size = options.get('vocab_size', 10000)
      self.oov_token = options.get('oov_token', '<OOV>')
      self.max_len = options.get('word_len', 100)
      self.padding_type = options.get('padding_type', 'post')
      self.truncation_type = options.get('truncation_type','post')
      self.embeddings_index = dict()
      self.categories = options.get('classes', ['Other', 'Feedback', 'Complaints'])
      self.len_categories = len(self.categories)
      self.embeddings_path = options.get('embeddings_path', os.path.join(self.model_path,'glove.6B.100d.txt'))
      self.train_path = options.get('train_data_path', os.path.join(self.model_path,'social_media_tracker_correct.csv'))
      self.test_path = options.get('test_data_path', os.path.join(self.model_path, 'social_media_tracker_correct.csv'))
      self.confusion_matrix_save_path = options.get('confusion_matrix_save_path', os.path.join(self.model_path,'confusion_matrix.jpg'))
      self.epochs = options.get('epochs', 5)
      self.batch_size = options.get('batch_size', 32)

      if os.path.splitext(self.embeddings_path)[1] != '.txt':
        raise FileNotFoundError('Specify the embeddings txt path')
      if os.path.splitext(self.train_path)[1] != '.csv':
        raise FileNotFoundError("Expecting the training data in a csv file")
      if os.path.splitext(self.test_path)[1] != '.csv':
        raise FileNotFoundError("Expecting the testing data in a csv file")

  def load_config(self,model_path):
    p = os.path.join(model_path, 'config.json')
    if os.path.exists(p):
      with open(p) as f:
        d = json.load(f)
      
      if os.path.splitext(d['model'])[1] != '.h5':
        raise FileNotFoundError("The model extension should be of .h5 format")
      self.model = d['model']
      if os.path.splitext(d['train_data_path'])[1] != '.csv':
        raise FileNotFoundError("The training file should be of .csv format")
      self.train_path = d['train_data_path']
      self.max_len = d['word_len']
      self.categories = d['classes']
      self.vocab_size = d['vocab_size']
      self.oov_token = d['oov_token']
      self.padding_type = d['padding_type']
      self.truncation_type = d['truncation_type']
      self.embeddings_path = d['embeddings_path']
      if not os.path.exists(d['embeddings_path']):
        raise FileNotFoundError("The embeddings file should be of .txt format")
      self.model_path = d['model_path']
      self.test_path = d['test_data_path']
      self.confusion_matrix_save_path = d['confusion_matrix_save_path']
      self.tokenizer = self.token(self.vocab_size, self.oov_token)
      self.epochs = d['epochs']
      return True
    else:
      return False
    return True

  def save_config(self, m, options):
    options['model'] = m
    options['oov_token'] = self.oov_token
    pdb.set_trace()
    out_file = open(os.path.join(self.model_path,'config.json'),'w')
    json.dump(options, out_file, indent=4)
    out_file.close()

  def token(self,vocab_size, oov_token):
    return Tokenizer(num_words=vocab_size,oov_token=oov_token)
    
  def data_formatting(self,df):
    df.loc[(df['category']!='Complaint') & (df['category']!='Feedback'), 'category'] = 'Other'
    df['text'] = df['text'].replace(numpy.nan,'None')
    return df
    
  def load_data(self,path):
    if os.path.splitext(path)[1] != '.csv':
      raise FileNotFoundError("Expecting the training data in a csv file")
    df = pd.read_csv(path)
    df = self.data_formatting(df)
    if df.columns.tolist() != (['text','category'] or ['category','text']):
      raise ValueError("The headers of the data frame are not correct. The headers should be in ['text','category'] format")
    return df['text'], df['category']

  def word_embeddings(self,word_index):
    f = open(self.embeddings_path)
    for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      self.embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(self.word_index)+1, self.max_len))
    for word, i in self.word_index.items():
        embedding_vector = self.embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

  def train_csv(self):
    x , y = self.load_data(self.train_path)
    return self.train(x,y)
  
  def train(self,x,y):
    '''
    Description:
    1) This module converts the raw texts into tokens and creates a sequence or series of tokens and is used to train the 
    Sequential model.
    2) This module trains the Sequential model designed to classify the text into Complaints or Feedback or Other.
    3) The Sequential model contains and embedding layer, two bidirectional LSTM layers and 2 dense layers.
    Inputs:
    The input to this module is text and its corresponding labels.
    Output:
    This module doesn't return anything but creates a trained Sequential model.
    '''
    x, y = self.load_data(self.train_path)
    if (type(x) != pandas.core.series.Series) or (type(y) != pandas.core.series.Series):
      raise TypeError("The data should be in the Data Frames format")
    self.tokenizer = self.token(self.vocab_size, self.oov_token)
    self.tokenizer.fit_on_texts(x)
    self.word_index = self.tokenizer.word_index
    x_sequence = self.tokenizer.texts_to_sequences(x)
    x_pad = pad_sequences(x_sequence,maxlen=self.max_len,padding=self.padding_type,truncating=self.truncation_type)
    
    if not self.ready_to_test:
      embedding_matrix = self.word_embeddings(self.word_index)

      embedding_layer = Embedding(input_dim=len(self.word_index)+1,
                              output_dim=self.max_len,
                              weights=[embedding_matrix],
                              input_length=self.max_len,
                              trainable=False)
      self.model = Sequential([
                      embedding_layer,
                      Bidirectional(LSTM(150,return_sequences=True)),
                      Bidirectional(LSTM(150)),
                      Dense(128, activation='relu'),
                      Dense(self.len_categories,activation='sigmoid')
                      ])
      self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    y = y.factorize()
    y = to_categorical(y[0], self.len_categories)
    x_train,x_test,y_train,y_test = train_test_split(x_pad,y,test_size=0.2,random_state=42)
    self.model.fit(x_train,y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_test,y_test))
    if not os.path.exists(self.model_path):
      os.makedirs(self.model_path)
    m = os.path.join(self.model_path, 'model.h5')
    self.model.save(m)
    self.save_config(m,self.options)

  def predict(self,y):
    '''
    Description:
    This module is used to predict the probabilities for each class i.e., complaint, feedback and other.
    Inputs:
    The raw text
    Outputs:
    The probabilies of each class.
    '''
    if type(y) != str:
      raise TypeError("The input should be a string")
    if self.ready_to_test:
      self.tokenizer.fit_on_texts(y)
      tw = self.tokenizer.texts_to_sequences(y)
      tw = pad_sequences(tw,maxlen=self.max_len)
      if isinstance(self.model,str):
        model = tensorflow.keras.models.load_model(self.model)
      prediction = model.predict(tw)[0]
      return {self.categories[k]: float(v) for k, v in enumerate(prediction)}
    else:
      raise ValueError('The test data is not ready')
  
  def predict_class(self,y):
    '''
    Description:
    This module is used to predict the class of the text based on the maximum probabilities.
    Inputs:
    The raw text
    Outputs:
    The predicted class based on the probabilies.
    '''
    prediction = self.predict(y)
    return max(prediction, key=prediction.get)
    
  def test_csv(self):
    x,y = self.load_data(self.test_path)
    return self.test(x,y)
    
  def test(self, x_test,y_test,display_metrics=False):
    y_pred = []
    y_test = list(y_test)
    for i in x_test:
      y_pred.append(self.predict_class(i))
    if display_metrics:
      self.display_metrics(y_test,y_pred)
    return accuracy_score(y_test,y_pred)

  def display_metrics(self,y_test,y_pred):
    mcm = multilabel_confusion_matrix(y_test,y_pred,labels=self.categories)
    if not os.path.exists(self.confusion_matrix_save_path):
      os.makedirs(self.confusion_matrix_save_path)
    plt.matshow(mcm)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(self.model_path, self.confusion_matrix_save_path))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"The accuracy of the model is: {accuracy}")
    print(f"The precision of the model is: {precision}")
    print(f"The recall of the model is: {recall}")
    print(f"The f1 score of the model is: {f1}")
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    opt = opts().parse()
    opt_dict = vars(opt)
    architecture = TextClassification(**opt_dict)
    if opt.task == 'train':
        architecture.train_csv()
    else:
        text = "I visited Auto Vista Vadki branch today. To my surprise I did not see anyone wearing masks or following Covid protocols. No sanitizers, No social distancing and serving customers on a daily basis. Are we really safe at Maruti Showrooms? I don't feel safe at all."
        #architecture.test_csv()
        print(architecture.predict(text))
