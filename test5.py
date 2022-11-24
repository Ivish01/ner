import csv
import requests
import json
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer

with open('C:/Users/Admin/Downloads/test_set_ran.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        
        r = requests.get("http://ner.iamdave.ai/tags?q={}".format(row[2]),
                            
                            headers={"Content-type": "application/json" } )
        
        r_dict = r.json()
        # print(r_dict)

    def evaluate(ner_model, r_dict):
        scorer = Scorer()
        for input_, annot in r_dict:
            doc_gold_text = ner_model.make_doc(input_)
            gold = GoldParse(doc_gold_text, entities=annot)
            pred_value = ner_model(input_)
            scorer.score(pred_value, gold)
        return scorer.scores