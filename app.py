import streamlit as st
import pandas as pd
import numpy as np
import ski-kit learn as sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

st.title("Hydraulic Rock Drill Fault Classification Machine Learning Model")

# Define the remove_outliers function
def remove_outliers(df, columns):
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Load the dataset
st.write("Loading the dataset...")
df = pd.read_csv('/Users/sarithavuppula/Downloads/Summer School/final_version_data.csv')

# Remove outliers
df = remove_outliers(df, ['pin', 'po', 'pdmp'])
st.write("Outliers removed.")

# Feature extraction with tqdm to track progress
def extract_features(df):
    features = []
    for _, group in tqdm(df.groupby(['condition', 'mode', 'cycle']), desc="Extracting features"):
        pin_mean = group['pin'].mean()
        po_mean = group['po'].mean()
        pdmp_mean = group['pdmp'].mean()
        pin_var = group['pin'].var()
        po_var = group['po'].var()
        pdmp_var = group['pdmp'].var()
        pin_skew = group['pin'].skew()
        po_skew = group['po'].skew()
        pdmp_skew = group['pdmp'].skew()
        pin_kurt = group['pin'].kurtosis()
        po_kurt = group['po'].kurtosis()
        pdmp_kurt = group['pdmp'].kurtosis()
        features.append([group['cycle'].iloc[0], group['condition'].iloc[0], group['mode'].iloc[0], 
                         pin_mean, po_mean, pdmp_mean, pin_var, po_var, pdmp_var, 
                         pin_skew, po_skew, pdmp_skew, pin_kurt, po_kurt, pdmp_kurt])
    return pd.DataFrame(features, columns=['cycle', 'condition', 'mode', 
                                           'pin_mean', 'po_mean', 'pdmp_mean', 
                                           'pin_var', 'po_var', 'pdmp_var', 
                                           'pin_skew', 'po_skew', 'pdmp_skew',
                                           'pin_kurt', 'po_kurt', 'pdmp_kurt'])

st.write("Extracting features...")
features_df = extract_features(df)
st.write("Feature extraction completed.")

# Normalize features
scaler = StandardScaler()
X = features_df.iloc[:, 3:]
y = features_df['mode']

X_scaled = scaler.fit_transform(X)

# Address class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
st.write("Data balanced with SMOTE.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Simplified Hyperparameter tuning for RandomForest with tqdm to track progress
param_grid = {
    'n_estimators': [100, 150],  # Reduced number of estimators
    'max_depth': [10, 15],       # Fewer depth options
    'min_samples_split': [2, 5], # Fewer options for splits
}

# Initialize GridSearchCV with fewer options and tqdm progress tracking
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=3,  # Reduced to 3 folds
                           n_jobs=-1, 
                           verbose=0)

st.write("Starting hyperparameter tuning...")
with tqdm(total=len(param_grid['n_estimators']) * len(param_grid['max_depth']) * 
          len(param_grid['min_samples_split']), desc="Hyperparameter Tuning") as pbar:
    grid_search.fit(X_train, y_train)
    pbar.update()

best_model = grid_search.best_estimator_

# Cross-validation for robustness
cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=3)
st.write(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.4f}")

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix for Fault Classification')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Print detailed classification report
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))
