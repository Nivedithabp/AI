###################################
# CS B551 Fall 2023, Assignment #3
#
# Your names and user ids:
#



import random
import math
from collections import defaultdict

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#


label_list = ['adv', 'noun', 'adp', 'prt', 'det', 'num', '.', 'pron', 'verb', 'x', 'conj', 'adj']

class Solver:
    def __init__ (self):
        self.initial_probability = dict()
        self.label_frequency = dict()
        self.transition_probability = defaultdict(dict)
        self.emission_probability = defaultdict(dict)
        self.final_probability = dict()
        
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            post = 0
            for idx, word in enumerate(sentence):
                tag = label[idx]
                if tag in self.emission_probability[word]:
                    post += math.log(self.emission_probability[word][tag])
            return post
        elif model == "HMM":
            post = 0
            for idx, word in enumerate(sentence):
                tag = label[idx]
                if tag in self.emission_probability[word]:
                    if idx == 0:
                        post += math.log(self.emission_probability[word][tag])+math.log(self.initial_probability[tag])
                    else: post += math.log(self.emission_probability[word][tag])+math.log(self.transition_probability[label[idx-1]][tag] if tag in self.transition_probability[label[idx-1]] and self.transition_probability[label[idx-1]][tag] != 0 else 1)
            return post

    def normalize(self):
        for text in self.emission_probability:
            for label in self.emission_probability[text]:
                self.emission_probability[text][label] /= self.label_frequency[label]
        
        for fl in self.transition_probability:
            for sl in self.transition_probability[fl]:
                self.transition_probability[fl][sl]/=self.label_frequency[fl]
                
        for label in self.initial_probability:
            self.initial_probability[label] /= self.label_frequency[label]
            
        for label in self.final_probability:
            self.final_probability[label] /= self.label_frequency[label]
    
    # Do the training!
    #
    def train(self, data):
        self.transition_probability = {i:{j:0 for j in label_list} for i in label_list}
        for text,label in data:
            for i in range(len(text)):
                if i == 0:
                    if label[i] not in self.initial_probability: self.initial_probability[label[i]] = 0
                    self.initial_probability[label[i]]+=1
                if i == len(text)-1:
                    if label[i] not in self.final_probability: self.final_probability[label[i]] = 0
                    self.final_probability[label[i]]+=1
                if text[i] in self.emission_probability:
                    if label[i] not in self.emission_probability[text[i]]: self.emission_probability[text[i]][label[i]] = 0
                    self.emission_probability[text[i]][label[i]] += 1
                else:
                    self.emission_probability[text[i]] = {label[i]:1}
                if label[i] in self.label_frequency: self.label_frequency[label[i]] +=1
                else: self.label_frequency[label[i]] =1
                
            for index in range(0, len(label) - 1):
                if label[index] not in self.transition_probability:
                    self.transition_probability[label[index]] = dict()
                else:
                    if label[index+1] not in self.transition_probability[label[index]]: self.transition_probability[label[index]][label[index+1]] = 0
                    self.transition_probability[label[index]][label[index+1]] += 1
        self.normalize()
        # print(self.transition_probability)
        # print(self.emission_probability)

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        predict_label_prob = [0] * len(sentence)
        predict_label = ["noun"]*len(sentence)
        
        for idx,word in enumerate(sentence):
            for text in self.emission_probability:
                if text == word:
                    for label in self.emission_probability[text]:
                        if predict_label_prob[idx]<self.emission_probability[text][label]:
                            predict_label[idx] = label
                            predict_label_prob[idx] = self.emission_probability[text][label]
        return predict_label


    def hmm_viterbi(self, sentence):
        viterbi_probability_matrix = [0] * len(sentence)
        predict_label = ["noun"]*len(sentence)
        
        for idx,word in enumerate(sentence):
            if word in self.emission_probability:
                for text in self.emission_probability:
                    if text == word:
                        for label in self.emission_probability[text]:
                            if idx == 0:
                                if viterbi_probability_matrix[idx]<self.emission_probability[text][label]*self.initial_probability[label]:
                                    predict_label[idx] = label
                                    viterbi_probability_matrix[idx] = self.emission_probability[text][label]*self.initial_probability[label]
                            elif viterbi_probability_matrix[idx]<self.emission_probability[text][label]*(self.transition_probability[predict_label[idx-1]][label] if label in self.transition_probability[predict_label[idx-1]] else 1):
                                predict_label[idx] = label
                                viterbi_probability_matrix[idx] = self.emission_probability[text][label]*(self.transition_probability[predict_label[idx-1]][label] if label in self.transition_probability[predict_label[idx-1]] else 1 )
            else:
                for label in self.transition_probability[predict_label[idx-1]]:
                    if viterbi_probability_matrix[idx]<self.transition_probability[predict_label[idx-1]][label]:
                        predict_label[idx] = label
                        viterbi_probability_matrix[idx] = self.transition_probability[predict_label[idx-1]][label]
        return predict_label



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")