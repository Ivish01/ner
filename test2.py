import csv
import requests
import json
from predectors import ner
from predectors import text_classification
import os
from flask import jsonify

uuid = []
with open('C:/Users/Admin/Downloads/test_set_ran.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        uuid.append(row)

    for i in uuid:
        filepath = os.path.join("MODEL_FOLDER", "W2V_MODEL_NAME")
        print(filepath)   

        # print(row)

        
        r = requests.get("http://ner.iamdave.ai/tags?q=sushant",
                            
                            headers={"Content-type": "application/json" } )
        
        r_dict = r.json()
        # print(r_dict)

        v=ner.Tagger()
        print( jsonify(v.predect(r)))


# uuid = []
# with open('C:/Users/Public/file.csv', 'r') as file:
#   reader = csv.reader(file)
#   for row in reader:
#     uuid.append(row)

# for i in uuid:
#   filepath = os.path.join("org/datasets/",  i , "/data")
#   print(filepath)


        
      

        

# for first 35 predicted words by this model is compared with the actual lable of csv file and got
# 19 correctly predicted and 16 incorrecly predicted 

# accuracy= 19/(19+16)
# print(accuracy)

# accuracy= 54.28

