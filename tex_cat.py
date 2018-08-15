# coding=utf-8
# use natural language toolkit
import codecs
import json
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import datetime
stemmer = LancasterStemmer()

class TexClassify():

    def __init__(self):
        self.training_data = []
        self.input_hidden_weight = []
        self.hidden_output_weight = []

        self.ERROR_THRESHOLD = 0.2

        self.words = []  # all of the words
        self.classes = []  # all of the classes
        self.documents = []  # all of the classes and text list

        self.training = []
        self.output = []

        self.load_setting()

    # load our calculated setting values
    def load_setting(self):
        print("Setting Loading")
        setting_file = 'settings.json'
        with open(setting_file) as data_file:
            weights = json.load(data_file)
            self.input_hidden_weight = np.asarray(weights['input_hidden_weight'])
            self.hidden_output_weight = np.asarray(weights['hidden_output_weight'])

    # initialize input neurons and output neurons number
    # prepare word list from all of the documents
    def get_word_class_space(self):
        ignore_words = ['?']
        for pattern in self.training_data:
            # cümle içindeki herbir kelimeyi tokenize etsin
            w = nltk.word_tokenize(pattern['sentence'])

            # words extends with join w
            self.words.extend(w)
            # add into documents with sentence words and sentence category
            self.documents.append((w, pattern['class']))

            if pattern['class'] not in self.classes:
                self.classes.append(pattern['class'])

        # kelime köküne iner
        self.words = [stemmer.stem(w.lower()) for w in self.words if w not in ignore_words]

        # Tekrar edilmeyen kelimeler listesi oluşturulur
        self.words = list(set(self.words))
        self.classes = list(set(self.classes))

        print("input summary")
        print (len(self.documents), "documents")
        print (len(self.classes), "classes")
        print (len(self.words), "unique stemmed words")

    # prepare input array by checking is the exists or no
    # if exist put 1 else put 0
    def prepare_dataset_as_input(self):
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            bag = []
            # herbir veriden alınan kök kelimeler
            pattern_words = doc[0]
            # herbir veriden alınan kök kelimelerin kökünü bul
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # Kelimelere göre 1 ve 0 lardan oluşan listeler yaıpılır
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # eğitim seti hazırlanıyor
            self.training.append(bag)
            output_row = list(output_empty)
            # classes.index(doc[1]) classes listesindeki class indexini döndürür
            # çıktı düğümleri eğitim setine göre hazırlanır
            output_row[self.classes.index(doc[1])] = 1
            self.output.append(output_row)

    def get_training_data(self):
        with codecs.open('business.txt', 'r', encoding='utf8') as f:
            for line in f:
                self.training_data.append({"class": "business", "sentence": line})

        with codecs.open('entertainment.txt', 'r', encoding='utf8') as f:
            for line in f:
                self.training_data.append({"class": "entertainment", "sentence": line})

        with codecs.open('politics.txt', 'r', encoding='utf8') as f:
            for line in f:
                self.training_data.append({"class": "politics", "sentence": line})

    # sigmoid fonksiyonu
    def sigmoid(self, x):
        output = 1 / (1 + np.exp(-x))
        return output

    # sigmoid fonsiyonun türevi
    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def sentence_tokenize_stemmer(self, sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    # return an array which is represents sentence words from data set array
    def get_word_array(self, sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.sentence_tokenize_stemmer(sentence)
        # word_number_array of words. There are zeros up to the number of words
        word_number_array = [0] * len(words)
        for sentence_word in sentence_words:
            # index is order, data_set_word is word
            for index, data_set_word in enumerate(words):

                if data_set_word == sentence_word:
                    word_number_array[index] = 1
                    if show_details:
                        print ("found in word_space: %s" % data_set_word)

        return (np.array(word_number_array))

    # layers computations
    def compute(self, sentence, show_details=False):
        # input list as 0 and 1
        word_numeric_array = self.get_word_array(sentence.lower(), self.words, show_details)
        # input layer is our array of words
        input_layer = word_numeric_array

        # calculation for hidden layer
        hidden_layer = self.sigmoid(np.dot(input_layer, self.input_hidden_weight))

        # calculation for output layer
        output_layer = self.sigmoid(np.dot(hidden_layer, self.hidden_output_weight))
        return output_layer

    def trainer(self, X, y, hidden_neurons=10, alpha=1, epochs=1000, dropout=False, dropout_percent=0.5):
        print ("Training with %s neurons in hidden layer, alpha:%s, dropout:%s %s" % (
            hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
        print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(self.classes)))
        np.random.seed(1)

        last_mean_error = 1
        # randomly initialize our weights with mean 0
        input_hidden_weight = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
        hidden_output_weight = 2 * np.random.random((hidden_neurons, len(self.classes))) - 1

        prev_input_hidden_weight_update = np.zeros_like(input_hidden_weight)
        prev_hidden_output_weight_update = np.zeros_like(hidden_output_weight)

        input_hidden_weight_direction_count = np.zeros_like(input_hidden_weight)
        hidden_output_weight_direction_count = np.zeros_like(hidden_output_weight)

        for j in iter(range(epochs + 1)):

            # Feed forward through layers 0, 1, and 2
            input_layer = X
            hidden_layer = self.sigmoid(np.dot(input_layer, input_hidden_weight))

            if (dropout):
                hidden_layer *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
                        1.0 / (1 - dropout_percent))

            output_layer = self.sigmoid(np.dot(hidden_layer, hidden_output_weight))

            # target error rate
            # y is target array
            error = y - output_layer

            if (j % 1000) == 0 and j > 500:
                # if this 10k iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(error)) < last_mean_error:
                    print ("delta after " + str(j) + " epochs:" + str(np.mean(np.abs(error))))
                    last_mean_error = np.mean(np.abs(error))
                else:
                    print ("break:", np.mean(np.abs(error)), ">", last_mean_error)
                    break

            # effect of output layer on error
            output_layer_delta = error * self.sigmoid_derivative(output_layer)

            # calculate effection of hidden layer weights on output layer
            hidden_layer_error = output_layer_delta.dot(hidden_output_weight.T)

            hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(hidden_layer)

            hidden_output_weight_update = (hidden_layer.T.dot(output_layer_delta))
            input_hidden_weight_update = (input_layer.T.dot(hidden_layer_delta))

            if (j > 0):
                input_hidden_weight_direction_count += np.abs(
                    ((input_hidden_weight_update > 0) + 0) - ((prev_input_hidden_weight_update > 0) + 0))
                hidden_output_weight_direction_count += np.abs(
                    ((hidden_output_weight_update > 0) + 0) - ((prev_hidden_output_weight_update > 0) + 0))

            hidden_output_weight += alpha * hidden_output_weight_update
            input_hidden_weight += alpha * input_hidden_weight_update

            prev_input_hidden_weight_update = input_hidden_weight_update
            prev_hidden_output_weight_update = hidden_output_weight_update

        now = datetime.datetime.now()

        # persist synapses
        settings = {'input_hidden_weight': input_hidden_weight.tolist(),
                    'hidden_output_weight': hidden_output_weight.tolist(),
                    'datetime': now.strftime("%Y-%m-%d %H:%M"),
                    'words': self.words,
                    'classes': self.classes
                    }
        setting_file = "settings.json"

        # setting is recorded into json file
        with open(setting_file, 'w') as outfile:
            json.dump(settings, outfile, indent=4, sort_keys=True)
        print ("saved synapses to:", setting_file)

    # test function
    def classify(self, sentence, show_details=False):
        results = self.compute(sentence, show_details)

        # i is index r is result
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        # getting classes from classes list by index
        return_results = [[self.classes[r[0]], r[1]] for r in results]
        print ("%s \n classification: %s" % (sentence, return_results))
        return return_results

    def main(self):
        self.get_training_data()
        self.get_word_class_space()
        self.prepare_dataset_as_input()




if __name__ == "__main__":
    classify = TexClassify()
    classify.main()

    # x is training set
    # y is output set
    X = np.array(classify.training)
    y = np.array(classify.output)

    # classify.trainer(X, y, hidden_neurons=20, alpha=0.1, epochs=10000, dropout=False, dropout_percent=0.2)

    text = raw_input("Enter sentence that you want to classify:\n")
    while text != 'quit':
        classify.classify(text, show_details=True)
        text = raw_input("Enter sentence that you want to classify:\n")



