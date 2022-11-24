import csv
import requests
import json
from predectors import ner
from predectors import text_classification
from predectors.text_classification import TextClassification
# from predectors import app
import os
from flask import jsonify
import pandas as pd

with open('C:/Users/Admin/Downloads/test_set_ran.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row[2])
        x=row[2]
        y=row[3]

        v = ner.Tagger()
        print(jsonify(v.predect('x')))
            
            # w=text_classification.TextClassification()
            # return w.display_matrix()

# df=pd.read_csv('C:/Users/Admin/Downloads/test_set_ran.csv')
# print(df.head())

# x=df['word']
# y=df['label']

# v = ner.Tagger()
# print(jsonify(v.predect(x)))