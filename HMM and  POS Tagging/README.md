## Part 1

In this problem, we are trying to predict the POS of a word in a given sentence. We have a training set of words and their POS.
We implement the POS tagging in 2 different ways:

- **Simple**
	
	Here we use the Bayes' Rule and calculate the probability of each word belonging to a specific POS tag. We can also look at this as a
	HMM without any transitions. The individiual words are the observed states and the POS are the hidden states. The initial and emission probabilities are calculated from the training corpus. 
	
	- The initial probabilities are calculated as - Probability of a certain POS being the POS of the first word in the sentence
	- The emission probabilities are calculated as - Probability of a word belonging to a certain POS (Number of times a word occurs as a certain POS/No of occurrences of that POS in corpus)

	We then use these to calculate probability of the word belonging to a certain POS and select the POS with greatest probability as the output
		

- **HMM**

	Using the Hidden Markov Model with the individual words as the observed states and the POS are the hidden states.
	
	- The initial probabilities are calculated as - Probability of a certain POS being the POS of the first word in the sentence
	- The emission probabilities are calculated as - Probability of a word belonging to a certain POS (Number of times a word occurs as a certain POS/No of occurrences of that POS in corpus)
	- We calculate the transition probabilities as - Probability of one POS following another POS (Number of times a certain POS follows another POS/Number of occurrences of that first POS(previous) in corpus)
	
	We then implement the Viterbi algorithm to find the MAP labelling for that sentence using the above probabilities.
	

Our results on bc.test


==> So far scored 2000 sentences with 29442 words.            
                   
		   Words correct:     Sentences correct: 
   
   0. Ground truth:      100.00%              100.00%
         
         1. Simple:       92.91%               43.15%
            
            2. HMM:       94.03%               48.95%
            
## Part 2

In this problem, we are performing optical character recognition. We use a training image and training dataset to learn the grid patterns of the characters in images and the probabilities of characters following each other.
We are trying to predict the text in the given test images using 2 different algorithms.

Here, once we get the grids of the test and train images, we compare them based on the average number '\*' pixels in the test and train image. Through this difference, we determine the noise of the character in the image and consequently adjust the emission probabilities for noise. 

- **Simple**

	In the simple algorithm, similar to simple model of Part1, we use Bayes' rule and consider an HMM with only initial and emission and no transition probabilities. The image characters are the hidden and observed states
	
	- The initial probabilities are calculated as - Probability of a certain character occuring as the character in the first grid in test image
	- The emission probabilities are calculated as- Probability of occurence of a character/probability that the grid represents a certain character
	
	We then select the most probable character in the test grid based on the probability of occurence of that character

- **HMM**
	We create a HMM with the image characters as the hidden and observed states.
	 
	- The initial probabilities are calculated as - Probability of a certain character occuring as the character in the first grid in test image
	- The emission probabilities are calculated as- Probability of occurence of a character/probability that the grid represents a certain character
	- We calculate the transition probabilities as - Probaility of one character following another (Number of times a character follows another character/Number of times the first/previous character occurs in the training corpus)
	
	We use Viterbi algorithm to find the most probable text for the test image using all of the -  certain character occuring as the character in the first grid, the probability of occurence of a character and the probability of a character following a previous character.


For our submission we have used the bc.train file from Part 1 as the training file

We have tried this solution on bigger datasets, and our conclusion is - the bigger the dataset and the more the training data, the better the accuracy.


As for our individual contributions to this assignment, we have all had discussions and worked together on researching, formulating, coding and debugging both parts.
