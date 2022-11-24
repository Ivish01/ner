import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.compat.v1.nn import rnn_cell as rnn, static_bidirectional_rnn
import os, re
from gensim.models import Word2Vec
from predectors import PredectionModel

W2V_MODEL_NAME=None
MODEL_FOLDER="./model"

class Tagger(PredectionModel):
    def __init__(self):
        super().__init__()

        self.model = Word2Vec.load(os.path.join(MODEL_FOLDER, W2V_MODEL_NAME))
        self.init_model()

    def get_vector(self, text, vector_size= 24):
        def _to_vector(s):
            l = []
            # print(len(s))
            for x in s:
                l.append(self.model.wv[x])
            return l
        def _yield_vectors(s):
            ind = 0
            if not s.strip():
                raise StopIteration
            s = s.strip().split()
            def _word(w, vector_size):
                w = " ".join(re.findall("[a-zA-Z]+", w)).lower()
                wrap_size = vector_size-len(w)-(vector_size-len(w))//2, (vector_size-len(w))//2
                return "{}{}{}".format(" "*wrap_size[0],w," "*wrap_size[1])
            
            for i in s:
                yield ind, _to_vector(_word(i, vector_size)), i
                ind+=1

        return _yield_vectors(text)

    def init_model(self):
        num_input = 25 # MNIST data input (img shape: 28*28)
        timesteps = 24 # timesteps
        num_hidden = 24 # hidden layer num of features
        num_classes = 4 # MNIST total classes (0-9 digits)

        X = tf.placeholder("float", [None, timesteps, num_input], "input")
        Y = tf.placeholder("float", [None, num_classes], "output")
        self.X = X
        self.Y = Y
        weights = {
            'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
        }

        biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        def BiRNN(x, weights, biases):

            x = tf.unstack(x, timesteps, 1)
            lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

            try:
                outputs, _, _ = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                    dtype=tf.float32)
            except Exception: # Old TensorFlow version only returns outputs not states
                outputs = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                dtype=tf.float32)

            return tf.matmul(outputs[-1], weights['out']) + biases['out']

        logits = BiRNN(X, weights, biases)
        self.prediction = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.vocab = ['country', 'location', 'other', 'person']
        
        self.input = tf.placeholder(dtype=tf.string, shape=(None,))
        matches = tf.stack([tf.equal(self.input, s) for s in self.vocab], axis=-1)
        self.onehot = tf.cast(matches, tf.float32)
    
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        #to write summary
        #writer1 = tf.summary.create_file_writer('log/model1')

        #with writer.as_default():
        #    tf.summary.text('Model configuration', s, step=0)
        
        self.sess.run(init)
        self.saver.restore(self.sess, os.path.join(MODEL_FOLDER, "lite"))

        print("sess", self.sess)
        print("saver", self.saver)
        print("logits", logits)
        print("prediction", self.prediction)
        print("self.accuracy", self.accuracy)
        print("self.input", self.input)
        print("correct_pred", correct_pred)
        print("init", init)
        print("matches", matches)

    def predect(self, text):
        pre = []
        # {
        #     Index: 0,
        #     Word: dinesh,
        #     Class:name,
        #     Probs:{name:0.5, location: 0.4, other: 0.1}
        # },
        lst_cls = ""
        index = 0 
        for ind, data, w in self.get_vector(text):
            ob = {
                "index": index,
                "word": w,
                "class": None,
                "probs": {}
            }

            
            for z in self.vocab:
                test_label = self.sess.run(self.onehot, feed_dict={self.input: [z]})
                acc = self.sess.run(self.accuracy, feed_dict={self.X: [data], self.Y: test_label})
                # print(z, test_label)
                if round(acc, 1) > 0.0:
                    ob["probs"][z] = float(round(acc, 1))
            cls = max(ob["probs"], key=ob["probs"].get)
            ob["class"] = cls
            
            if lst_cls and lst_cls != "other" and lst_cls == cls:
                pre[-1]["word"] = "{} {}".format(pre[-1]["word"], w)

                prbs = dict( (k, ( pre[-1]["probs"].get(k, 0.0)+ ob["probs"].get(k, 0.0))/2) for k in set( list(pre[-1]["probs"].keys()) + list(ob["probs"].keys() )))
                pre[-1]["probs"] = prbs
            else:
                pre.append(ob)
                index += 1
            lst_cls = cls
        """
        for i, pt in self.patterns.items():
            patt = pt
            canonical_fun = None
            formater_fun = None
            if isinstance(pt, dict):
                patt = pt["pattern"]
                canonical_fun = pt.get("canonical")
                formater_fun = pt.get("formater")

            if formater_fun:
                text = formater_fun(text)

            ls = re.finditer("(^|\s){}(\s|$)".format(patt), text)
            for t in ls:
                word = text[t.span()[0]:t.span()[1]]
                word = word[1:] if word[0] == " " else word
                canonical = None
                if canonical_fun:
                    canonical = canonical_fun(word)
                print(word, canonical)
                fl = False
                for z in pre:
                    if z["word"] == canonical or z["word"] == word:
                        z["probs"][i] = 1.0
                        fl= True
                        z["class"] = max(z["probs"], key=z["probs"].get)
                        z["canonical"] = canonical
                        break

                if not fl:
                    ind = ind + 1
                    ob = {
                        "index": ind,
                        "word": word,
                        "class": i,
                        "probs": {},
                        "canonical": canonical
                    }

                    ob["probs"][i] = 1.0
                    pre.append(ob)
                """
        return pre