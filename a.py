import csv
import requests
import json
import os
import pandas as pd

def flat_accuracy(text, annotations):
        actual_label = [annotations[3]]
        pred_label=[i['class'] for i in text]
        return  1 if actual_label == pred_label else 0

list_ones = []
with open('/home/ubuntu/ner/test_set_ran.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        
        r = requests.get("http://ner.iamdave.ai/tags?q={}".format(row[2]),
                            
                            headers={"Content-type": "application/json" } )
        
        r_dict = r.json()
#         print(r_dict)

        list_ones.append(flat_accuracy(r_dict,row))

predict_points = sum(list_ones)
accuracy = (predict_points)/9085 * 100
print(accuracy)
