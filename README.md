# Final-Project Bank Marketing

## Project/Goals
Project Title: Predicting Term Deposit Subscriptions in Bank Marketing Campaigns

Overview
This project focuses on the analysis of direct marketing campaigns conducted by a Portuguese banking institution through phone calls. The primary goal is to employ machine learning techniques to predict whether a client will subscribe to a term deposit. The dataset encompasses detailed records from May 2008 to November 2010, providing a comprehensive foundation for building predictive models.

Problem Definition
Clarify the Objective: Your project aims to predict whether clients will subscribe to a term deposit based on information gathered from direct marketing campaigns by a Portuguese bank.

Goals
- Data Understanding: Explore and understand the dataset, consisting of various inputs related to client demographics, campaign details, and economic indicators that may influence a client's decision to subscribe to a term deposit.
- Predictive Modeling: Utilize machine learning algorithms to accurately predict the outcome of a client subscribing to a term deposit. This involves preprocessing data, selecting features, and evaluating different models to find the most effective approach.
- Model Evaluation: Assess the performance of the models using appropriate metrics such as accuracy, precision, recall, and the AUC-ROC curve, to ensure the reliability and effectiveness of the predictions.
- Insights and Recommendations: Derive insights from the analysis to understand key factors influencing term deposit subscriptions. Provide recommendations to the banking institution on optimizing their marketing strategies based on predictive analytics.## Process

### (your step 1)
Project Initialization and Data Preprocessing Steps
In the initial phase of our data science project aimed at predicting term deposit subscriptions for a Portuguese banking institution, we undertook a structured approach to understand and prepare our dataset for further analysis and modeling. Below is a detailed breakdown of the steps involved:

1. Data Discovery and Initial Understanding
We commenced our project by conducting a thorough search of the dataset to familiarize ourselves with its structure and contents. This preliminary exploration was crucial for gaining insights into the characteristics of the data, including the identification of the total number of features present. Our dataset comprises various attributes that detail client demographics, campaign interactions, and economic indicators, setting the foundation for our analysis.

2. Hypothesis Generation
Following the initial data exploration, we embarked on generating hypotheses to guide our analysis. This step involved formulating assumptions based on domain knowledge and initial observations, which would later be tested through statistical and machine learning models. Hypothesis generation is pivotal in directing our exploratory data analysis (EDA) and feature selection strategies.

3. EDA and Data Cleaning
Our exploratory data analysis revealed that the dataset is highly imbalanced and notably free of null or NaN values, indicating a well-maintained but skewed dataset. The imbalance in the dataset underscores the challenge in predicting term deposit subscriptions, necessitating careful consideration in model selection and evaluation metrics.

4. In-depth EDA for Key Variables
We conducted a focused EDA on specific variables of interest, including pdays, duration, age, job, education, and loan, to unravel their distributions and potential impact on the target variable. This step is essential for understanding the dynamics between client characteristics and their subscription behaviors.

5. Pairwise Feature Exploration
Utilizing pair plots, we visualized the relationships between all features within the dataset. This graphical representation helped in identifying patterns, trends, and correlations among the variables, providing valuable insights for feature engineering and model development.

6. Feature Reduction
To streamline our analysis, we removed features deemed less relevant or redundant, specifically dropping poutcome, contact, day, and month columns. This decision was based on preliminary findings that suggested these variables had minimal influence on the prediction of term deposit subscriptions.

7. Feature Renaming
We refined the dataset by renaming the target variable y to subscription, enhancing the clarity and interpretability of our data.

8. Handling Unknown Values
In our quest to maintain a high-quality dataset, we identified and removed approximately 2,018 records with unknown values, reducing the dataset size from 45,211 to 43,193 entries. This step ensures that our models are trained on more reliable and accurate data.

9. Correlation Analysis
Finally, we employed a heatmap to observe the correlations between various predictor variables and the target variable. This visual tool facilitated the identification of strongly and weakly correlated features, guiding our feature selection process for model training.


### (your step 2)Preprocessing & Feature Engineering

PART II: Preprocessing and Feature Engineering
In the second phase of our project, we focused on preprocessing and feature engineering to prepare our dataset for effective modeling. This phase is crucial for enhancing model performance by ensuring that the features are appropriately scaled, distributions are normalized, and categorical variables are suitably encoded. Here's a professional summary of the steps taken:

Scaling Numerical Features
To address the varying scales of our numerical features, we employed StandardScaler from Scikit-learn to standardize these features, excluding the target variable (subscription). This standardization process normalizes the feature values around a mean of 0 with a standard deviation of 1, ensuring that no single feature dominates the model due to its scale.

Normalizing Skewed Distributions
For features exhibiting skewed distributions, we applied the Yeo-Johnson transformation, which effectively normalizes these distributions, including those with zero and negative values. This step is vital for models that assume normality of the input features.

Encoding Categorical Variables
To convert categorical variables into a format suitable for machine learning models, we performed one-hot encoding on columns such as job, marital, and education. This process transforms categorical variables into a set of binary variables, each representing a category in the original feature.

Converting Binary Variables
Binary variables represented as "Yes" or "No" were mapped to 1 and 0, respectively, and their data types were converted to integers. This conversion ensures compatibility with our machine learning models, which require numerical input.

Preparing and Splitting the Dataset
With preprocessing complete, we prepared the dataset for modeling by separating the features from the target variable (subscription). We then shuffled the data to eliminate any underlying ordering or patterns. Subsequently, the data was split into training and testing sets using an 80:20 ratio, with 80% allocated for training to provide a substantial dataset for learning, and 20% reserved for testing to evaluate model performance.

Addressing Class Imbalance
Recognizing the class imbalance in our target variable, we calculated and normalized class weights for the binary classes. This approach ensures that our model pays more attention to the minority class during training, improving its ability to generalize and predict both classes accurately.

These preprocessing and feature engineering steps are critical in laying a solid foundation for the subsequent modeling phase. By meticulously scaling, normalizing, encoding, and preparing our data, we enhance the potential for our predictive models to yield accurate and reliable results, ultimately aiming to improve the effectiveness of the bank's marketing strategies for term deposit subscriptions.


### (your step 3) Training & Evaluating ML model
PART III: Model Training and Evaluation
Model Development Phase
Upon completing the data preprocessing and feature engineering stages, we transitioned to the core of our project: model training and evaluation. Our strategy involved experimenting with a diverse array of machine learning algorithms to identify the most effective model for predicting term deposit subscriptions. The algorithms selected for this comparative analysis were:

Logistic Regression
Support Vector Machine (SVM)
Decision Tree
K-Nearest Neighbors (KNN)
Naive Bayes
Random Forest
Each model was constructed using its default parameters as defined by Scikit-learn, establishing a baseline for performance comparison. This approach allowed us to assess the fundamental capabilities of each algorithm without the influence of extensive hyperparameter tuning.

Evaluation Criteria
The primary metric for model evaluation was accuracy, which measures the proportion of correctly predicted outcomes out of all predictions made. This metric provides a straightforward means of comparing the effectiveness of each algorithm in classifying clients' likelihood to subscribe to a term deposit.

Model Performance Results
The accuracy scores obtained from the baseline models are as follows:

Logistic Regression: Accuracy of 89.34%
Random Forest: Accuracy of 89.51%
Support Vector Machine (SVM): Accuracy of 89.14%
K-Nearest Neighbors (KNN): Accuracy of 88.27%
Decision Tree: Accuracy of 85.11%
Naive Bayes: Accuracy of 83.16%
Conclusion and Best Performing Model
Upon comparing the performance of each model, the Random Forest algorithm emerged as the best performer with an accuracy of 89.51%. This result underscores the Random Forest model's robustness and effectiveness in handling the predictive challenges of our dataset, making it the preferred choice for further optimization and deployment.


Enhanced Model Training and Evaluation with Random Forest
Overview
In refining our approach to predicting term deposit subscriptions, we employed the RandomForestClassifier from Scikit-learn, incorporating class weights to address the dataset's imbalance. This section outlines the steps taken to optimize the Random Forest model, including class weight calculation, model fitting, hyperparameter tuning, and evaluation.

Class Weight Adjustment
To mitigate the impact of class imbalance, we calculated and normalized class weights for the binary target variable. This process ensures that the model adequately learns from both classes, improving its predictive performance on the minority class.

Class Weights Calculation: We assigned class weights of 1.0 for class 0 and 10.0 for class 1, reflecting the importance of accurately predicting the minority class. These weights were normalized to ensure their sum equals 1, providing a balanced influence on the model training process.
Random Forest Classifier Implementation
With the class weights adjusted, we instantiated the RandomForestClassifier, specifying the normalized class weights and a fixed random state for reproducibility. The model was then trained on our preprocessed training dataset, encompassing both numerical and categorical features prepared in the previous phases.

Model Evaluation
The trained Random Forest model's performance was evaluated using accuracy as the primary metric, alongside a detailed classification report to assess its precision, recall, and F1-score across both classes.

Accuracy Metrics: The initial model demonstrated promising accuracy, laying the groundwork for further optimization through hyperparameter tuning.
Hyperparameter Tuning
To enhance the model's performance, we conducted hyperparameter tuning using RandomizedSearchCV, exploring a vast parameter space to identify the optimal configuration.

Search Space: Parameters such as n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, and criterion were varied to explore their impact on model accuracy.
Optimization Results: The tuning process identified an optimal set of hyperparameters that further improved the model's accuracy, demonstrating the effectiveness of Random Forest in handling this predictive task.
Confusion Matrix Visualization
The final step in our evaluation involved generating and visualizing a confusion matrix for the optimized Random Forest model. This visualization provided insights into the model's performance, specifically its ability to distinguish between the two classes (subscribers vs. non-subscribers).

Insights: The confusion matrix highlighted the model's precision in predictions, showcasing its strength in identifying true positives and true negatives while minimizing false positives and false negatives.

the resultFinal Analysis and Business Impact of the Predictive Model
Overview of Dataset Characteristics
Through meticulous analysis, we observed the following key characteristics within our dataset, which provided a foundational understanding of the bank's clientele and their interactions:

Age Distribution: The age of clients ranges from 18 to 95 years, with an average age of 41 years, indicating a broad spectrum of financial maturity and needs.
Bank Balance: Balances range significantly from -€8,019 to €102,127, with an average balance of €1,362, highlighting varied financial standings among clients.
Call Durations: The duration of calls varied from 0 to 4,918 seconds, averaging at 258 seconds, suggesting diverse levels of engagement and interest.
Campaign Contacts: The number of contacts per campaign ranged from 1 to 63, with an average of 3 calls, pointing to varied persistence levels in reaching out to clients.
Pdays: Days since the last contact range from -1 (never contacted) to 871, with an average of 40 days, indicating different follow-up strategies.
Previous Contacts: Prior contacts range from 0 to 275, with an average of 0.58, showing differing levels of historical engagement.
Client Demographics: Notably, 21.5% of clients work in blue-collar jobs, 60% are married, 51% have secondary education, and 98% have no credit defaults.
Subscription Rates: 88% of clients did not subscribe to a term deposit following the campaign, highlighting the challenge at hand.
Dataset Challenges Addressed
Our analysis identified and addressed several critical challenges within the dataset:

Skewed Distributions: Attributes such as bank balance exhibited skewed distributions, which were normalized for more effective analysis.
Class Imbalance: The target variable, indicating subscription to a term deposit, showed significant imbalance, effectively managed through class weight adjustments in our Random Forest Classifier model.
Model Performance and Results
The culmination of our efforts resulted in a Random Forest model that:

Generalized Effectively: Achieved an impressive 89% accuracy on test data, demonstrating strong predictive capability.
Improved Through Tuning: Benefited from hyperparameter tuning, which marginally enhanced its accuracy, solidifying its robustness.
Business Implications and Transformative Impact
The predictive model's success heralds transformative potential for the bank's marketing strategy:

Enhanced Subscription Rates: Previously, the campaign's conversion rate was approximately 11.62%. With the model's 89% accuracy, we anticipate a significant jump in successful subscriptions, potentially reaching a conversion rate of around 89%.
Strategic Customer Targeting: The model enables precise identification of clients likely to subscribe, optimizing marketing efforts and resource allocation.
Elevated Return on Investment (ROI): By focusing on high-potential clients, the bank can significantly increase ROI, reducing the costs associated with reaching uninterested parties.
Boosted Morale and Brand Perception: Success in these initiatives not only enhances employee morale through more fruitful interactions but also bolsters the bank's image, fostering a perception of understanding and meeting customer needs.
In conclusion, our data-driven approach, underpinned by a rigorously developed and tuned Random Forest model, sets the stage for a marked enhancement in the bank's marketing efficacy. By leveraging predictive analytics, the bank is poised to redefine its engagement strategy, yielding substantial benefits in efficiency, customer satisfaction, and financial performance.


FLASK MODEL : 

For the bank deposit classification project, a Flask web application was developed, comprising three key components to facilitate deployment and user interaction:

Model Files: Contains the serialized Random Forest model (bank_deposit_classification.pkl), enabling the application to leverage pre-trained insights for predictions.

Templates: Hosts HTML files for the web interface, allowing users to input data and view predictions through a user-friendly platform.

App: The app.py script acts as the core of the application, managing routes, processing user inputs, and integrating the model to deliver real-time predictions.

This Flask application transforms the bank deposit classification model into an interactive tool, providing a practical interface for users to obtain predictions based on client data. It exemplifies an efficient approach to deploying machine learning models for accessible and immediate decision support.

## Challenges 
Many challenges were faced during this project, such as:

Data Imbalance: The target variable, indicating whether a client subscribed to a term deposit, was significantly imbalanced. This imbalance posed a risk of biasing the model towards the majority class, potentially undermining its predictive accuracy for the minority class.

Skewed Feature Distributions: Several features, such as bank balance and call durations, exhibited skewed distributions. This skewness could impact the model's ability to learn effectively, necessitating techniques like transformation and normalization to mitigate its effects.

Feature Selection and Engineering: With numerous features available, determining the most relevant ones for predicting term deposit subscriptions required extensive analysis. This process involved balancing the inclusion of informative features against the risk of overfitting or introducing noise.

Hyperparameter Tuning Complexity: The process of hyperparameter tuning, especially for the Random Forest model, was computationally intensive and time-consuming. Finding the optimal set of parameters involved navigating a vast parameter space, requiring a balance between thoroughness and computational efficiency.

Handling Missing and Unknown Data: The dataset contained missing and unknown values, particularly in categorical features. Deciding on the best approach to handle these values without introducing bias or losing valuable information was a critical challenge.

Model Evaluation and Validation: Given the imbalanced nature of the dataset, standard accuracy metrics were not sufficient to fully evaluate model performance. Employing additional metrics and validation techniques to ensure the model's generalizability and effectiveness was crucial.

Computational Resources: Some steps, particularly hyperparameter tuning and processing large datasets, demanded significant computational resources. Efficiently managing these resources while ensuring comprehensive model training and evaluation was a logistical challenge.


