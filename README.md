# Medicare_Insurance_Fraud_Detection

This project is to detect potential fraud in Medicare claims by analyzing healthcare provider data and applying machine learning models to classify fraudulent and non-fraudulent claims. The dataset contains information on providers, claims, reimbursements, and other related details.

## Project Overview

	•	Type: Machine Learning Classification
	•	Goal: Predict whether a provider has committed fraud based on claim data.
	•	Key Steps:
      	1.	Exploratory Data Analysis (EDA)
      	2.	Data Preprocessing
      	3.	Feature Selection/Extraction
      	4.	Model Training
      	5.	Evaluation

## Dataset

The dataset used in this project has the following structure:

- **Rows** : 5,410 entries
- **Columns** : 158 features related to providers, claim amounts, and patient conditions.

**Some key columns include:**

	•	Provider: Unique identifier for each provider.
	•	PotentialFraud: Target variable indicating fraud (Yes/No).
	•	InscClaimAmtReimbursed: Total amount reimbursed for the claim.
	•	DeductibleAmtPaid: Amount paid towards the deductible.
	•	Various columns representing the number of claims per provider, condition details, and more.

## Data Preprocessing

**Several preprocessing steps were performed:**

	•	Missing Data Handling: Missing values were checked and appropriately handled.
	•	Feature Engineering: New features were derived, such as counts of claims by provider and diagnostic codes.
	•	Data Transformation: Continuous features were scaled, and categorical variables were encoded.

**Exploratory Data Analysis (EDA)**

EDA was conducted to understand the distribution of data and identify patterns related to fraud:

	•	Summary statistics were generated.
	•	Key features influencing fraud were visualized.
	•	Correlation analysis was performed to detect feature relationships.

**Feature Selection**

Features were selected based on domain knowledge and statistical analysis. The final feature set includes:

	•	Claim-based features: Count of claims, claim amounts, diagnosis and procedure codes.
	•	Provider-based features: Number of claims by the provider and interactions with physicians.

## Model Training

Several machine learning models were trained and evaluated, including:

	•	Logistic Regression
	•	SVM

**Evaluation**

From the two notebooks, the following metrics were extracted for the fraud detection models:

**File: MEDICARE_FRAUD_EDA**

	•	Percent distribution of potential fraud:
	•	No Fraud: 61.88%
	•	Fraud: 38.12%

The following are the performance metrics from two different models:

**First Model: Logistic Regression**
	•	Accuracy: 90% <br>
	•	Confusion Matrix:<br>
&emsp;&emsp;&emsp;•	True Negatives (TN): 1353<br>
&emsp;&emsp;&emsp;•	False Positives (FP): 118<br>
&emsp;&emsp;&emsp;•	False Negatives (FN): 39<br>
&emsp;&emsp;&emsp;•	True Positives (TP): 113<br>
  •	Precision (class 0): 0.97, Precision (class 1): 0.49<br>
  •	Recall (class 0): 0.92, Recall (class 1): 0.74<br>
  •	F1-score (class 0): 0.95, F1-score (class 1): 0.59<br>
  •	Weighted F1-score: 0.91<br>
  •	ROC AUC: 0.83<br>
 
**Second Model: SVM**
	•	Accuracy: 94%<br>
	•	Confusion Matrix:<br>
&emsp;&emsp;&emsp;•	True Negatives (TN): 1459<br>
&emsp;&emsp;&emsp;•	False Positives (FP): 12<br>
&emsp;&emsp;&emsp;•	False Negatives (FN): 92<br>
&emsp;&emsp;&emsp;•	True Positives (TP): 60<br>
	•	Precision (class 0): 0.94, Precision (class 1): 0.83<br>
	•	Recall (class 0): 0.99, Recall (class 1): 0.39<br>
	•	F1-score (class 0): 0.97, F1-score (class 1): 0.54<br>
	•	Weighted F1-score: 0.93<br>
	•	ROC AUC: 0.69<br>

## Final Analysis:

	•	First Model (ROC AUC = 0.83, Accuracy = 90%) has a better balance between precision and recall for detecting fraudulent cases.
	•	Second Model (ROC AUC = 0.69, Accuracy = 94%) has a higher accuracy but performs worse in recall for fraud detection, indicating that it misses more fraudulent cases (lower recall for class 1).

Thus, the first model is better at identifying fraud, though the second model has a slightly higher overall accuracy. ￼

## Files in this Repository

	•	MEDICARE_FRAUD_EDA.ipynb: Notebook containing the exploratory data analysis and feature engineering.
	•	MEDICARE_TEST.ipynb: Notebook where models were trained and tested, and final evaluations were performed.

## How to Run

	1.	Clone the repository.
	2.	Install necessary dependencies using the requirements.txt file.
	3.	Run the EDA notebook for exploratory analysis.
	4.	Execute the model training notebook to train and evaluate models.

## Future Work

	•	Tuning models further using hyperparameter optimization.
	•	Incorporating more complex models like neural networks.
	•	Investigating additional data sources for improving fraud detection accuracy.
