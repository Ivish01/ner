import csv
import requests
import json
from predectors import ner
from predectors import text_classification
import os
from flask import jsonify
import spacy
from spacy.training import Example
from spacy.scorer import Scorer

with open('C:/Users/Admin/Downloads/test_set_ran.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        
        r = requests.get("http://ner.iamdave.ai/tags?q=sushant",
                            
                            headers={"Content-type": "application/json" } )
        
        r_dict = r.json()
        # print(r_dict)

        nlp = spacy.load("path_to_your_model")
        scorer = Scorer()

        examples = []
        annots = []

        for content in r_dict:
            predicted = nlp(content['row'])
            for annotate_content in content['annotation']:
                # start = annotate_content['start']
                # end = annotate_content['end']
                label = annotate_content['label']
                annots.append(label)
   
            final = {'entities': annots}
    
            example = Example.from_dict(predicted, final)
            examples.append(example)

        scores = scorer.score(examples)
        scores['ents_per_type']






















        # v=ner.Tagger()
        # v.predect(r_dict)
    # print( jsonify(v.predect(r)))
