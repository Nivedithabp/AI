## Part 1 : Raichu

The original starter code does not contain any logic to find the next best state as expected in the question. In our approach, we are using the Min-Max algorithm with alpha-beta pruning in this two-player game to maximize the chance of our player winning. We generate the possible successors of each Pichu, Pikachu, and Raichu on the board at each move and assign a score to each of the successors based on a scoring function.

**Problem Formulation**

**Initial State**: An NxN grid, the state of the board given in the input.

**To-Move**: The player given in the input.

### **Actions**:

- **Pichu**: 
  - Move forward diagonally by 1 space.
  - Jump over an opponent Pichu and move forward diagonally 2 spaces.

- **Pikachu**: 
  - Move left, right, or forward by 1 or 2 spaces.
  - Jump over opponent Pichu and Pikachu and move forward left or right by 2 or 3 spaces.

- **Raichu**: 
  - Move forward or backward any number of spaces as long as the path is empty.
  - Jump over opponent Pichu and Pikachu and Raichu as long as there are no other pieces between.

### **Successor Function**: 
- The NxN grid results after performing any of the valid actions (moves) on any of the pieces of the current player.

### **Utility Function/Scoring Function**:

For the current player, the weight for Pichu is 20, Pikachu is 50, and Raichu is 500. For the opponent, Pichu is 20, Pikachu is 50, and Raichu is 700, respectively, with Raichu being valued higher than our raichu. Other factors considered in the scoring function are:

- Material Weight: Piece weight on the board.
- Safety Weight: Pichu and Pikachu safety.
- Safety is determined by checking if possible spaces above (black) or below (white) players are empty for Pichu and Pikachu, and if Raichu is on the edges of the board.
- Mobility score is added whether the piece is moving forward toward becoming Raichu, based on index increased or decreased.
- Is at an edge: Probability of Pikachu movement is double than that of Pichu.
- Points for becoming a Raichu, considering the index position and possible attacks as well as whether it is in corners (0,0), (0,n-1), (n-1,0), (n-1,n-1), which are safe for Raichu and have more points, each with different values.

The final score for the board is calculated as follows: 

`final_score = 3 * material weight + 2.25 * safety weight + 1.25 * Mobility + 2.25 * points for becoming a Raichu`

Values given are computed after a lot of trial and error. We return `final_score - depth * 0.1` for each successor, with 0.1 * depth as a scaling function to avoid overestimating the score.

**Is-Terminal/Goal State**: The goal state is when there are no opponent pieces left on the board or when there is a time constraint with the best possible move for past depth totally calculated, and more pieces than the opponent on the board.

**Descriptions for some of our functions**:

- **Is safe**: Determines if a piece at a given location is safe from any opponent attack.

- **has_player_won**: Determines whether there are no opponent pieces left (the same function is used for both players).

- **can_I_attack**: Returns a dictionary with the count of pieces the opponent can attack. For example, it returns `{b: 0, B: 1, $: 2}`.

- **scoring_function**: Evaluation function for the minimax algorithm, which follows the utility function described above.

To explain our approach, we are running a Min-Max algorithm and generating a search tree, using alpha-beta pruning to avoid visiting successors that will not lead to the optimal solution. After generating and scoring our successors, we select the successor with the highest score during our turn and the successor with the lowest score during the opponent's turn. When there are equal scores, we randomly select the moves. The Min-Max algorithm is run to the allowed depth within the given time limit, and we select the best move from the explored tree within that time.

## Part2 - Truth be Told
The provided code is a Python program that implements a Naive Bayes classifier for text classification. Here's a summary of the code:

1. Loading Data:
   - The code starts by loading training and testing data from text files.
   - Each line in the files represents a text object with a label. ( deceptive, truthful )
   - The training data contains two lists: one for text objects and one for their corresponding labels.
   - The code also extracts the unique classes (labels) present in the data.

2. Naive Bayes Classifier:
   - The `classifier` function is the core of the Naive Bayes classifier.
   - It calculates the class prior probabilities (P(class)) and word probabilities (P(word | class) based on the training data.
   - The class data is separated to calculate class prior probabilities.
   - Laplace smoothing is used for handling unknown words. To improve the future it has added lower cases alphabets
   - The function then classifies text objects in the test data by computing the log-likelihood of each class for each text object and selecting the class with the highest likelihood.

3. Main Function:
   - It loads and sanitizes the data to prepare it for classification.
   - The classifier is applied to the test data, and the predicted labels are compared with the actual labels to calculate classification accuracy.

4. Output:
   - The code outputs the classification accuracy as a percentage.

This code is designed to classify text objects into two categories using a Naive Bayes classifier. It loads training and testing data, trains the classifier on the training data, and then uses the trained classifier to predict labels for the test data, evaluating its performance by calculating classification accuracy.
