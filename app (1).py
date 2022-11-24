
from flask import Flask, request,jsonify

from predectors.text_classification import TextClassification
from predectors import ner
import json, os

app = Flask(__name__)

SERVER_NAME = os.environ.get('SERVER_NAME', '127.0.0.1:5052')
W2V_MODEL_NAME = os.environ.get('W2V_MODEL_NAME', "train_model.bin")
HTTP = os.environ.get('HTTP', 'http')
app.config['PREFERRED_URL_SCHEME'] = HTTP
app.config['MODEL_FOLDER'] = 'model'

#setting up default model names
ner.MODEL_FOLDER = app.config['MODEL_FOLDER']
ner.W2V_MODEL_NAME = W2V_MODEL_NAME

@app.route("/health")
def health_check():
    return "OK", 200

#gives the list of  all the trained models
#initialize the text classifier with the path of each model and store it in the dictionary where the key is the name of the model and the value is the initialized instance of the Text_Classifier

#import pdb
#pdb.set_trace()

#TEXT_CLASSIFIERS = {
#    "maruti_voc": TextClassification(opts_for_testing)
#}

# f = open('./model/text_classification_trained_models/config.json')
# data = json.load(f)

# TEXT_CLASSIFIERS = {
#     "maruti_voc": TextClassification(**data)
# }
# text_classifier_path = os.environ.get('TEXT_CLASSIFIERS', './')
# for model in os.listdir(text_classifier_path):
#     with open(os.path.join(text_classifier_path, model, 'config.json')) as f:
#       j = json.load(f)
#       TEXT_CLASSIFIERS[model] = TextClassification(model_path = os.path.join(text_classifier_path, model), **j)
@app.route('/tags/<model_name>')
def text(model_name):
    if model_name not in TEXT_CLASSIFIERS:
      return jsonify(error=f"Model {model_name} not found")
    arg = request.args.get("q")
    if not arg:
        return jsonify({})
    return jsonify(TEXT_CLASSIFIERS[model_name].predict(arg))

        
@app.route("/tags")
def p():
    arg = request.args.get("q")
    if not arg:
        return jsonify({})
    v = ner.Tagger()
    return jsonify(v.predect(arg))




if __name__ == "__main__":
    # rl = role.Role('Admin')
    # rl.create_update("SaleTxn", {'customer_id': 'C00041', 'station_code': 'BWA', 'fuel_name': "DIESEL"}, auth_key="asdfasdf")
    app.run(debug=False, host="0.0.0.0", port=5052)
