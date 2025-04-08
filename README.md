# ELE489_HW2_21947358

## Banknote Authentication with Decision Tree
This project applies a Decision Tree Classifier on the UCI Banknote Authentication dataset to classify banknotes as authentic or fake using statistical features extracted from images of the banknotes.

### Dataset
 
The dataset consists of 1,372 instances, each with 4 numerical features:

- Variance of the wavelet-transformed image

- Skewness of the wavelet-transformed image

- Kurtosis of the wavelet-transformed image

- Entropy of the image

Each instance is labeled as:

- 0: Fake

- 1: Authentic

Dataset used: [UCI Banknote Authentication Data Set](https://archive.ics.uci.edu/dataset/267/banknote+authentication)



### What This Project Does
- Loads and visualizes the dataset using Seaborn pairplots

- Splits data into training and testing sets (80/20)

- Trains a DecisionTreeClassifier using different hyperparameters

- Evaluates the model using accuracy, precision, recall, and F1-score

- Displays the confusion matrix

- Visualizes the decision tree structure

- Shows feature importance scores




### Requirements
- Make sure the following Python libraries are installed:

pip install pandas matplotlib seaborn scikit-learn




### How to Run

- Download the dataset file and create a python project. Place the dataset file data_banknote_authentication.txt in the python project directory. 


### Notes
- The classifier uses the entropy criterion with a max_depth of 4 and min_samples_split of 10.

- You can modify these parameters to observe how they affect accuracy and tree complexity.

- The visualizations help understand feature importance and how decisions are made in the tree.



### Sample Output
- Classification report with precision, recall, and F1-score

- Confusion matrix

- Decision tree plot

- Feature importance plot













