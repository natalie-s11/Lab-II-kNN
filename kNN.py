# %%
# Import libraries from last lab.
# Imports - Libraries needed for data manipulation and ML preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %%
# Step 1:

# Review datasets
College = pd.read_csv("college_completion.csv")
College.head()
# Question: Can I predict if a college is private based on the state where
# the college is located?


# %%
# Use def function from previous lab.
def prep_college_pipeline(college_path):
    College = pd.read_csv(college_path)

    # Change data types
    College['cohort_size'] = College['cohort_size'].fillna(0).astype(int)

    College['hbcu'] = College['hbcu'].apply(
        lambda x: True if x == 1 else False)
    College['flagship'] = College['flagship'].apply(
        lambda x: True if x == 1 else False)

    College['is_four_year'] = College['level'].apply(
        lambda x: True if x == 'four-year' else (
            False if x == 'two-year' else pd.NA)
    ).astype('boolean')

    # Drop level column
    College.drop(['level'], axis=1, inplace=True, errors='ignore')

    # One-hot encode
    College = pd.get_dummies(College, columns=['control'], prefix='control')

    # Scaling
    scaler = MinMaxScaler()
    College['cohort_size_scaled'] = scaler.fit_transform(
        College[['cohort_size']])
    College['student_count_scaled'] = scaler.fit_transform(
        College[['student_count']])

    # Drop original columns after scaling
    College.drop(['cohort_size', 'student_count'], axis=1, inplace=True)

    # Drop columns
    College.drop([
        'long_x', 'lat_y', 'site', 'awards_per_value', 'awards_per_state_value',
        'awards_per_natl_value', 'exp_award_value', 'exp_award_state_value',
        'exp_award_natl_value', 'exp_award_percentile', 'ft_pct', 'fte_value',
        'fte_percentile', 'vsa_grad_elsewhere_after6_first', 'vsa_enroll_after6_first',
        'vsa_enroll_elsewhere_after6_first', 'vsa_grad_after4_transfer',
        'vsa_grad_elsewhere_after4_transfer', 'vsa_enroll_after4_transfer',
        'vsa_enroll_elsewhere_after4_transfer', 'vsa_grad_after6_transfer',
        'vsa_grad_elsewhere_after6_transfer', 'vsa_enroll_after6_transfer',
        'vsa_enroll_elsewhere_after6_transfer', 'similar', 'state_sector_ct',
        'carnegie_ct', 'counted_pct', 'nicknames', 'vsa_year',
        'vsa_grad_after4_first', 'vsa_grad_elsewhere_after4_first',
        'vsa_enroll_after4_first', 'vsa_enroll_elsewhere_after4_first',
        'vsa_grad_after6_first'
    ], axis=1, inplace=True, errors='ignore')

    # Define the target variable
    College['private_school'] = (
        College['control_Private for-profit'] +
        College['control_Private not-for-profit']
    )

    # Train / Tune / Test Split
    train, temp = train_test_split(
        College,
        train_size=0.55,
        stratify=College['private_school'],
        random_state=42
    )

    tune, test = train_test_split(
        temp,
        train_size=0.5,
        stratify=temp['private_school'],
        random_state=42
    )

    # Specify what to return
    return train, tune, test


# %%
# Call the function to prepare the data with the appropriate input.
college_train, college_tune, college_test = prep_college_pipeline(
    "college_completion.csv")

# %%
# Confirm the splits based on the question initially asked
print("College splits:")
print(college_train.shape, college_tune.shape, college_test.shape)
print(college_train['private_school'].value_counts(normalize=True))


# %%
# Step 2: Build a kNN model to predict your target variable using 3
# nearest neighbors. Make sure it is a classification problem, meaning if
# needed changed the target variable.


# Keep ONLY state column as the feature
X_train = college_train[['state']]
y_train = college_train['private_school']

X_test = college_test[['state']]
y_test = college_test['private_school']

print("Features being used:")
print(X_train.columns.tolist())
print(f"\nNumber of features: {len(X_train.columns)}")

# One-hot encode state
X_train_encoded = pd.get_dummies(X_train, columns=['state'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=['state'], drop_first=True)

# Make sure train and test have the same columns
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, join='left', axis=1, fill_value=0)

print(f"\nFeature matrix shape after encoding:")
print(f"Training: {X_train_encoded.shape}")
print(f"Test: {X_test_encoded.shape}")

# Build the kNN classifier with k=3
knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn_model.fit(X_train_encoded, y_train)

# Make predictions
y_pred = knn_model.predict(X_test_encoded)

# Check the results
print(f"\nModel trained successfully!")
print(f"Number of predictions: {len(y_pred)}")
print(f"Prediction distribution:")
print(pd.Series(y_pred).value_counts(normalize=True))


# %%
# Step 3: Create a dataframe that includes the test target values, test
# predicted values, and test probabilities of the positive class.

# Get probabilities (probability of class 0 and class 1)
y_pred_proba = knn_model.predict_proba(X_test_encoded)

# Create the dataframe
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    # Probability of class 1 (private school)
    'probability_positive_class': y_pred_proba[:, 1]
})

print(results_df.head(10))
print(f"\nDataFrame shape: {results_df.shape}")


# %%
# Step 4: If you adjusted the k hyperparameter what do you think would happen to the threshold function? Would the confusion matrix look the same at the same threshold levels or not? Why or why not?
# The confusion matrix would not look the same at the same threshold
# levels. This is because changing k changes the probability estimates.
# When the probabilities change, the same threshold will classify
# instances differently, which changes the confusion matrix values.


# %%
# Step 5: Evaluate the results using the confusion matrix. Then "walk"
# through your question, summarize what concerns or positive elements do
# you have about the model as it relates to your question?


# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\n")

# Label the confusion matrix for clarity
cm_df = pd.DataFrame(
    cm,
    index=['Actual Public (0)', 'Actual Private (1)'],
    columns=['Predicted Public (0)', 'Predicted Private (1)']
)
print("Labeled Confusion Matrix:")
print(cm_df)
print("\n")

# Calculate metrics
print("Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Public', 'Private']))
print("\n")

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.3f}")
print("\n")

# Break down the confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (correctly predicted public): {tn}")
print(f"False Positives (predicted private, actually public): {fp}")
print(f"False Negatives (predicted public, actually private): {fn}")
print(f"True Positives (correctly predicted private): {tp}")


# Based on my results, you can not really predict if a college will be
# public or private based on the state where the college is located.

# Positive Elements
# The model's accuracy was over 50%, which is better than random guessing in a binary classification problem.
# The model correctly predicted 371 private schools out of 504 True Positives.

# Concerns
# Something that I noticed was that when using all features (excepy the ones I dropped), the accuracy was much higher, at 83% so maybe I should have switched around my question.
# Just using state location alone is not a good predictor of whether a
# college is private or public.

# %%
# Step 6: Create two functions: One that cleans the data & splits into
# training|test and one that allows you to train and test the model with
# different k and threshold values, then use them to optimize your model
# (test your model with several k and threshold combinations). Try not to
# use variable names in the functions, but if you need to that's fine. (If
# you can't get the k function and threshold function to work in one
# function just run them separately.)


# Function 1: Clean data and split into training/test (FLEXIBLE VERSION)
def clean_and_split_data(file_path, target_col,
                         feature_cols, test_size=0.45, random_state=42):
    """
    Cleans data and splits into training and test sets.
    """
    # Read data
    df = pd.read_csv(file_path)

    # Prepare data (simplified version of prep_college_pipeline)
    df['cohort_size'] = df['cohort_size'].fillna(0).astype(int)
    df['hbcu'] = df['hbcu'].apply(lambda x: 1 if x == 1 else 0)
    df['flagship'] = df['flagship'].apply(lambda x: 1 if x == 1 else 0)

    # FIXED: Use correct values '4-year' and '2-year' (not 'four-year' and
    # 'two-year')
    df['is_four_year'] = df['level'].apply(
        lambda x: 1 if str(x).strip(
        ) == '4-year' else (0 if str(x).strip() == '2-year' else None)
    )

    df.drop(['level'], axis=1, inplace=True, errors='ignore')
    df = pd.get_dummies(df, columns=['control'], prefix='control')

    scaler = MinMaxScaler()
    df['cohort_size_scaled'] = scaler.fit_transform(df[['cohort_size']])
    df['student_count_scaled'] = scaler.fit_transform(df[['student_count']])
    df.drop(['cohort_size', 'student_count'], axis=1, inplace=True)

    df.drop([
        'long_x', 'lat_y', 'site', 'awards_per_value', 'awards_per_state_value',
        'awards_per_natl_value', 'exp_award_value', 'exp_award_state_value',
        'exp_award_natl_value', 'exp_award_percentile', 'ft_pct', 'fte_value',
        'fte_percentile', 'vsa_grad_elsewhere_after6_first', 'vsa_enroll_after6_first',
        'vsa_enroll_elsewhere_after6_first', 'vsa_grad_after4_transfer',
        'vsa_grad_elsewhere_after4_transfer', 'vsa_enroll_after4_transfer',
        'vsa_enroll_elsewhere_after4_transfer', 'vsa_grad_after6_transfer',
        'vsa_grad_elsewhere_after6_transfer', 'vsa_enroll_after6_transfer',
        'vsa_enroll_elsewhere_after6_transfer', 'similar', 'state_sector_ct',
        'carnegie_ct', 'counted_pct', 'nicknames', 'vsa_year',
        'vsa_grad_after4_first', 'vsa_grad_elsewhere_after4_first',
        'vsa_enroll_after4_first', 'vsa_enroll_elsewhere_after4_first',
        'vsa_grad_after6_first'
    ], axis=1, inplace=True, errors='ignore')

    # Create private_school column (in case it's the target)
    df['private_school'] = (
        df['control_Private for-profit'] + df['control_Private not-for-profit']
    )

    # Remove rows with missing target values
    df = df[df[target_col].notna()].copy()
    df[target_col] = df[target_col].astype(int)

    # Select features and target
    X = df[feature_cols]
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # One-hot encode categorical columns
    categorical_cols = X_train.select_dtypes(
        include=['object']).columns.tolist()
    X_train_encoded = pd.get_dummies(
        X_train, columns=categorical_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(
        X_test, columns=categorical_cols, drop_first=True)

    # Align columns
    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded, join='left', axis=1, fill_value=0
    )

    return X_train_encoded, X_test_encoded, y_train, y_test

# Function 2: Train and test model with different k and threshold values


def train_and_evaluate_knn(X_train, X_test, y_train,
                           y_test, k_value=3, threshold=0.5):
    """
    Trains a kNN model and evaluates it with a custom threshold.

    Parameters:
    - X_train: training features
    - X_test: test features
    - y_train: training target
    - y_test: test target
    - k_value: number of neighbors for kNN (default 3)
    - threshold: probability threshold for classification (default 0.5)

    Returns:
    - Dictionary with model performance metrics
    """
    # Train the model
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train, y_train)

    # Get probabilities
    # Probability of positive class
    y_pred_proba = knn.predict_proba(X_test)[:, 1]

    # Apply custom threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Return results
    return {
        'k': k_value,
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


# Test the functions with different k and threshold combinations
print("Testing different k and threshold combinations:\n")

# Prepare data using Function 1
X_train_enc, X_test_enc, y_train, y_test = clean_and_split_data(
    file_path="college_completion.csv",
    target_col='private_school',
    feature_cols=['state'],  # Using only state
    test_size=0.45,
    random_state=42
)

# Test different combinations
k_values = [3, 5, 7, 10, 15, 20]
threshold_values = [0.3, 0.4, 0.5, 0.6, 0.7]

results_list = []

for k in k_values:
    for thresh in threshold_values:
        result = train_and_evaluate_knn(
            X_train_enc, X_test_enc, y_train, y_test, k, thresh)
        results_list.append(result)
        print(f"k={k:2d}, threshold={thresh:.1f} -> "
              f"Accuracy: {result['accuracy']:.3f}, "
              f"Precision: {result['precision']:.3f}, "
              f"Recall: {result['recall']:.3f}, "
              f"F1: {result['f1_score']:.3f}")

# Create a summary dataframe
results_df = pd.DataFrame(results_list)
results_df = results_df.drop(
    ['predictions', 'probabilities', 'confusion_matrix'], axis=1)

print("\n" + "=" * 60)
print("SUMMARY OF ALL COMBINATIONS:")
print("=" * 60)
print(results_df.to_string(index=False))

# Find best combination by accuracy
best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
print(f"\n{'=' * 60}")
print("BEST MODEL BY ACCURACY:")
print(f"{'=' * 60}")
print(f"k = {best_accuracy['k']}")
print(f"threshold = {best_accuracy['threshold']}")
print(f"Accuracy = {best_accuracy['accuracy']:.3f}")
print(f"Precision = {best_accuracy['precision']:.3f}")
print(f"Recall = {best_accuracy['recall']:.3f}")
print(f"F1 Score = {best_accuracy['f1_score']:.3f}")

# Find best combination by F1 score
best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
print(f"\n{'=' * 60}")
print("BEST MODEL BY F1 SCORE:")
print(f"{'=' * 60}")
print(f"k = {best_f1['k']}")
print(f"threshold = {best_f1['threshold']}")
print(f"Accuracy = {best_f1['accuracy']:.3f}")
print(f"Precision = {best_f1['precision']:.3f}")
print(f"Recall = {best_f1['recall']:.3f}")
print(f"F1 Score = {best_f1['f1_score']:.3f}")

# %%
# Step 7: How well does the model perform? Did the interaction of the
# adjusted thresholds and k values help the model? Why or why not?

# My model does not preform well. My model would have been better if I used more features.
# The best accuracy I got was 0.609, 60.9%, when k=15 and threshold = 0.5 which was only slightly better than randomly guessing.
# Yes, adjusting thresholds and k values did help the model because it
# increased the accuracy. However, it was not the best thing ever since
# state is a weak predictor of if a college is public or private based on
# what state it is in.


# %%
# Step 8: Choose another variable as the target in the dataset and create another kNN model using the two functions you created in step 6.
# New Question: Can we predict if a college is a four-year institution
# based on state?


# Use Function 1 from Step 6 with a DIFFERENT target
X_train_fy, X_test_fy, y_train_fy, y_test_fy = clean_and_split_data(
    file_path="college_completion.csv",
    target_col='is_four_year',  # DIFFERENT TARGET!
    feature_cols=['state'],  # Just use state (like the original model)
    test_size=0.45,
    random_state=42
)

print(f"Training set: {X_train_fy.shape}")
print(f"Test set: {X_test_fy.shape}")
print(f"Number of features: {X_train_fy.shape[1]} (after encoding)")
print("\n")

# Use Function 2 from Step 6 to test different k and threshold combinations
print("Testing k and threshold combinations:\n")

k_values_fy = [3, 5, 10, 15]
threshold_values_fy = [0.3, 0.5, 0.7]

results_list_fy = []

for k in k_values_fy:
    for thresh in threshold_values_fy:
        result = train_and_evaluate_knn(
            X_train_fy, X_test_fy, y_train_fy, y_test_fy, k, thresh)
        results_list_fy.append(result)
        print(f"k={k:2d}, threshold={thresh:.1f} -> "
              f"Accuracy: {result['accuracy']:.3f}, "
              f"Precision: {result['precision']:.3f}, "
              f"Recall: {result['recall']:.3f}, "
              f"F1: {result['f1_score']:.3f}")

# Create summary
results_df_fy = pd.DataFrame(results_list_fy)
results_df_fy = results_df_fy.drop(
    ['predictions', 'probabilities', 'confusion_matrix'], axis=1)

print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
print(results_df_fy.to_string(index=False))

# Find best model
best_accuracy_fy = results_df_fy.loc[results_df_fy['accuracy'].idxmax()]
print(f"\n{'=' * 60}")
print("BEST MODEL:")
print(f"{'=' * 60}")
print(f"k = {best_accuracy_fy['k']}")
print(f"threshold = {best_accuracy_fy['threshold']}")
print(f"Accuracy = {best_accuracy_fy['accuracy']:.3f}")
print(f"Precision = {best_accuracy_fy['precision']:.3f}")
print(f"Recall = {best_accuracy_fy['recall']:.3f}")
print(f"F1 Score = {best_accuracy_fy['f1_score']:.3f}")

print("\n" + "=" * 60)
print("COMPARISON:")
print("=" * 60)
print(f"Original model (predict private/public from state):    60.9%")
print(
    f"New model (predict four-year/two-year from state):     {best_accuracy_fy['accuracy'] * 100:.1f}%")

# %%
