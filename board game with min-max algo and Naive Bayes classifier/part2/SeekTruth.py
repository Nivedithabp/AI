# SeekTruth.py : Classify text objects into two categories
#
# PLEASE PUT YOUR NAMES AND USER IDs HERE
#

import sys
import collections
import math

def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#

def classifier(train_data, test_data):
    # Initialize the Naive Bayes classifier parameters
    class_prior = {}  # P(class)
    word_probs = {}   # P(word | class)
    
    # Separate the training data into different classes
    class_data = collections.defaultdict(list)
    for i in range(len(train_data["objects"])):
        label = train_data["labels"][i]
        text = train_data["objects"][i].lower()
        class_data[label].append(text)
    
    # Calculate P(class)
    total_samples = len(train_data["labels"])
    for label, data in class_data.items():
        class_prior[label] = len(data) / total_samples
    
    # Calculate P(word|class)
    for label, data in class_data.items():
        words = " ".join(data).split()
        word_counts = collections.Counter(words)
        total_words = sum(word_counts.values())
        word_probs[label] = {word: (count + 1) / (total_words + len(word_counts)) for word, count in word_counts.items()}
    
    # Perform classification on the test data
    predicted_labels = []
    for text in test_data["objects"]:
        max_class = None
        max_score = -math.inf
        words = text.split()
        
        for label, prior in class_prior.items():
            score = math.log(prior)
            for word in words:
                if word in word_probs[label]:
                    score += math.log(word_probs[label][word])
                else:
                    # Handle unknown words with Laplace smoothing
                    score += math.log(1 / (total_words + len(word_counts)))

            if score > max_score:
                max_score = score
                max_class = label

        predicted_labels.append(max_class)
    
    return predicted_labels

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")
    # print("Training Data:")
    # print("Number of Objects:", len(train_data["objects"]))
    # print("Number of Labels:", len(train_data["labels"]))
    # print("Classes:", train_data["classes"])

    # print("\nTesting Data:")
    # print("Number of Objects:", len(test_data["objects"]))
    # print("Number of Labels:", len(test_data["labels"]))
    # print("Classes:", test_data["classes"])  
    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results= classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
