#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)


from PIL import Image, ImageDraw, ImageFont
import sys
from collections import defaultdict
import math
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!

# Parse and process data
def process_data (fname):
    file = open(fname, 'r')
    processed_string=[]
    for line in file:
        no_space = line
        for i,word in enumerate(no_space):
            if i%2==0:
                processed_string.append(word)
    file.close()
    return processed_string


training_data = process_data(train_txt_fname)

def train(data):
    initial_probability = dict()
    emission_probability = {i:{j:0 for j in range(len(test_letters))}for i in train_letters}
    transition_probability = defaultdict(dict)
    char_probability = dict()
    total_pixels = CHARACTER_HEIGHT*CHARACTER_WIDTH
    for letter in train_letters:
        for idx in range(len(test_letters)):
            letter_pix,test_pix = train_letters[letter],test_letters[idx]
            match = 0
            for r in range(CHARACTER_HEIGHT):
                for c in range(CHARACTER_WIDTH):
                    if letter_pix[r][c] == '*' and letter_pix[r][c] == test_pix[r][c]:
                        match+=1
            noise = 0.42
            prob = (match)/total_pixels
            fin = prob*(1-noise) * (noise*(1-prob))
#             print(fin)
            if fin>0.005: emission_probability[letter][idx] = fin
            
    for word in data:
        for idx,alpha in enumerate(word):
            if alpha not in char_probability: char_probability[alpha]= 0
            char_probability[alpha]+=1
            if idx == 0:
                if alpha not in initial_probability: initial_probability[alpha] = 0
                initial_probability[alpha]+=1
            else:
                if word[idx-1] not in transition_probability: transition_probability[word[idx-1]] = dict()
                if alpha not in transition_probability[word[idx-1]]: transition_probability[word[idx-1]][alpha] = 0
                transition_probability[word[idx-1]][alpha]+=1
    for i in transition_probability:
        for j in transition_probability[i]:
            transition_probability[i][j] /= char_probability[i]

    char_freq = sum(list(char_probability.values()))
    for i in initial_probability:
        initial_probability[i]/= len(data)
        
    for i in char_probability:
        char_probability[i]/=char_freq
    
    return initial_probability,emission_probability,transition_probability,char_probability

init,emit,transition,char_occ_prob = train(training_data)
# Transition Probability

def simple():
    y = [' ' for i in range(len(test_letters))]
    y_prob = [0 for i in range(len(test_letters))]
    for i in range(len(test_letters)):
        for j,letter in enumerate(train_letters):
            if y_prob[i]<(emit[letter][i]):
                y_prob[i] = emit[letter][i]
                y[i] = letter
    return ''.join(y)

def hmm():
    y = [' ' for i in range(len(test_letters))]
    y_prob = [0 for i in range(len(test_letters))]
    for i in range(len(test_letters)):
        for j,letter in enumerate(train_letters):
            if (i == 0) or (i>0 and y[i-1] == ' '):
                if y_prob[i]<emit[letter][i]:
                    y_prob[i] = emit[letter][i]
                    y[i] = letter
            elif y_prob[i]<emit[letter][i]*(transition[y[i-1]][letter] if y[i-1] in transition and letter in transition[y[i-1]] else 1):
                y_prob[i] = emit[letter][i]*(transition[y[i-1]][letter] if y[i-1] in transition and letter in transition[y[i-1]] else 1)
                y[i] = letter
    return ''.join(y)

# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print("\n".join([ r for r in train_letters['I'] ]))
# print("\n".join([ r for r in train_letters['1'] ]))
# # Same with test letters. Here's what the third letter of the test data
# #  looks like:
# print("\n".join([ r for r in test_letters[0] ]))



# The final two lines of your output should look something like this:
print("Simple: " + simple())
print("   HMM: " + hmm()) 


