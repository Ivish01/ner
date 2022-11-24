import csv
import requests
import json

with open('C:/Users/Admin/Downloads/test_set_ran.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        
        r = requests.get("http://ner.iamdave.ai/tags?q={}".format(row[2]),
                            
                            headers={"Content-type": "application/json" } )
        
        r_dict = r.json()
        print(r_dict)

        


# for first 35 predicted words by this model is compared with the actual lable of csv file and got
# 19 correctly predicted and 16 incorrecly predicted 

# accuracy= 19/(19+16)
# print(accuracy)

# accuracy= 54.28

