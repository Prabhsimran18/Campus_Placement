# Campus_Placement
1. Dataset Description and Preprocessing Steps
Dataset Description:
The dataset used for this project is the "Campus Recruitment Prediction" dataset from Kaggle. It contains various features related to students' academic performance, specialization, and work experience, with the target variable being whether the student was placed in a job or not. The dataset includes the following features:
sl_no: anonymous id unique to a given employee
gender: employee gender
ssc_p: SSC is Secondary School Certificate (Class 10th). ssc_p is the percentage
of marks secured in Class 10th.
ssc_b: SSC Board. Binary feature.
hsc_p: HSC is Higher Secondary Certificate (Class 12th). hsc_p is the percentage
of marks secured in Class 12th.
hsc_b: HSC Board. Binary feature.
hsc_s: HSC Subject. Feature with three categories.
degree_p: percentage of marks secured while acquiring the degree.
degree_t: branch in which the degree was acquired. Feature with three categories.
workex: Whether the employee has some work experience or not. Binary feature.
etest_p: percentage of marks secured in the placement exam.
specialisation: the specialization that an employee has. Binary feature.
mba_p: percentage of marks secured by an employee while doing his MBA.
status: whether the student was placed or not. Binary Feature. Target variable.
salary: annual compensation at which an employee was hired.


Preprocessing Steps:
Label Encoding: Applied to binary categorical variables (ssc_b, hsc_b, workex, specialisation, status).
One-Hot Encoding: Applied to categorical variables with multiple categories (hsc_s, degree_t).
Data Scaling: Applied StandardScaler to the features to ensure convergence for Logistic Regression.
Handling Missing Values: No missing values were present in the dataset. Since the model's objective is to forecast a student's likelihood of being placed—and only placed students receive salaries—the salary column was removed.


2. Models Selected and Rationale
Models Selected:
Logistic Regression: A simple and interpretable model for binary classification.
Decision Tree Classifier: A non-linear model that can capture complex relationships in the data.
Random Forest Classifier: An ensemble model that improves upon the decision tree by reducing overfitting.
Support Vector Machine (SVM): A powerful model for classification, especially with non-linear kernels.
Rationale:
Logistic Regression: Suitable for binary classification and provides well-calibrated probabilities.
Decision Tree Classifier: Capable of capturing non-linear relationships and is easy to interpret.
Random Forest Classifier: Combines multiple decision trees to improve accuracy and reduce overfitting.
Support Vector Machine (SVM): Effective for high-dimensional spaces and can handle non-linear relationships with appropriate kernels.

5. Conclusions
Best Model:
Support Vector Machine (SVM) performed the best with an accuracy of 0.8769, precision of 0.8913, recall of 0.9318, F1 score of 0.9111, and ROC-AUC of 0.8469.
Voting Classifier:
The Voting Classifier (Hard Voting) achieved an accuracy of 0.8154, precision of 0.8077, recall of 0.9545, F1 score of 0.8750, and ROC-AUC of 0.7392.
While the Voting Classifier performed well, it did not outperform the SVM model in terms of accuracy and F1 score.

Explanation of SVM Performance
The Support Vector Machine (SVM) model performed exceptionally well in this dataset due to several key factors:
1. Effective Handling of Non-Linear Relationships
Kernel Trick: SVMs can handle non-linear relationships between features and the target variable using kernel functions (e.g., linear, polynomial, radial basis function (RBF)). The RBF kernel, in particular, is effective in capturing complex patterns in the data.
Hyperplane Optimization: SVMs aim to find the optimal hyperplane that maximizes the margin between different classes. This optimization process helps in achieving better generalization and improved performance on unseen data.
2. Robust to Overfitting
Regularization: SVMs have a regularization parameter (C) that controls the trade-off between achieving a low error on the training data and minimizing the norm of the weights (maximizing the margin). A well-tuned C parameter can prevent overfitting, making the model more robust to noise and irrelevant features.
Feature Scaling: SVMs are sensitive to the scale of the input features. By applying StandardScaler, we ensure that all features are on a similar scale, which helps the SVM algorithm to converge faster and find the optimal hyperplane more effectively.
3. High-Dimensional Data Handling
Effective in High-Dimensional Spaces: SVMs are particularly effective in high-dimensional spaces, which is often the case after one-hot encoding categorical variables. The model can efficiently handle the increased dimensionality without suffering from the curse of dimensionality.
4. Balanced Precision and Recall
Precision-Recall Trade-off: SVMs can achieve a good balance between precision and recall, as evidenced by the high F1 score (0.9111). This balance is crucial for classification tasks where both false positives and false negatives are important.
Confusion Matrix Analysis: The confusion matrix for SVM shows a low number of false negatives and false positives, indicating that the model is making accurate predictions for both classes.
5. ROC-AUC Score
Strong Discriminative Power: The high ROC-AUC score (0.8469) indicates that the SVM model has a strong ability to distinguish between the positive and negative classes across different probability thresholds. This is a strong indicator of the model's overall performance.
Summary of SVM Performance
Accuracy: 0.8769
Precision: 0.8913
Recall: 0.9318
F1 Score: 0.9111
ROC-AUC: 0.8469
Confusion Matrix:
[[16,5],[3,41]]
Conclusion
The SVM model's strong performance can be attributed to its ability to handle non-linear relationships, robust regularization, effective feature scaling, and balanced precision-recall trade-off. These factors contribute to its high accuracy, precision, recall, F1 score, and ROC-AUC, making it the best-performing model for this dataset.
Recommendation
Given the performance metrics and the explanation of SVM's strengths, the Support Vector Machine (SVM) is recommended as the best model for predicting campus recruitment outcomes. The SVM model's strong performance in accuracy, precision, recall, and ROC-AUC makes it the most reliable choice for this classification task.
