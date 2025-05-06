# loan_prediction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pickle
import warnings
warnings.filterwarnings('ignore')

class LoanPredictionSystem:
    def __init__(self):
        self.df = None
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }
        self.best_model = None
        self.scaler = StandardScaler()
    
    def load_data(self, filepath):
        """Step 2: Load and clean dataset"""
        self.df = pd.read_csv(filepath)
        print("Dataset loaded successfully")
        print(f"Shape: {self.df.shape}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        # Basic cleaning
        self.df.drop(['Loan_ID'], axis=1, inplace=True)  # Remove unnecessary column
        
        # Fill missing values
        self._handle_missing_values()
        
        # Encode categorical variables
        self._encode_categorical()
        
        return self.df
    
    def _handle_missing_values(self):
        """Handle missing values according to feature nature"""
        # Numerical features - fill with median
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Categorical features - fill with mode
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
    
    def _encode_categorical(self):
        """Encode categorical variables to numerical"""
        # Gender (Male: 1, Female: 0)
        self.df['Gender'] = self.df['Gender'].map({'Male': 1, 'Female': 0})
        
        # Married (Yes: 1, No: 0)
        self.df['Married'] = self.df['Married'].map({'Yes': 1, 'No': 0})
        
        # Education (Graduate: 1, Not Graduate: 0)
        self.df['Education'] = self.df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
        
        # Self_Employed (Yes: 1, No: 0)
        self.df['Self_Employed'] = self.df['Self_Employed'].map({'Yes': 1, 'No': 0})
        
        # Property_Area (Ordinal encoding)
        self.df['Property_Area'] = self.df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
        
        # Dependents (Convert to numerical)
        self.df['Dependents'] = self.df['Dependents'].replace('3+', 3)
        self.df['Dependents'] = pd.to_numeric(self.df['Dependents'])
    
    def visualize_data(self):
        """Step 3: Data Visualization"""
        # Plot distribution of each feature
        plt.figure(figsize=(15, 20))
        for i, col in enumerate(self.df.columns[:-1]):  # Exclude target column
            plt.subplot(4, 3, i+1)
            sns.histplot(self.df[col], kde=True)
            plt.title(col)
        plt.tight_layout()
        plt.show()
        
        # Boxplot to check for outliers
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=self.df.drop('Loan_Status', axis=1))
        plt.title("Boxplot of Features (Before Outlier Removal)")
        plt.xticks(rotation=45)
        plt.show()
    
    def remove_outliers(self):
        """Step 4: Remove outliers using IQR method"""
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap the outliers
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
        
        # Visualize after outlier removal
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=self.df.drop('Loan_Status', axis=1))
        plt.title("Boxplot of Features (After Outlier Removal)")
        plt.xticks(rotation=45)
        plt.show()
    
    def check_multicollinearity(self):
        """Step 5: Check for multicollinearity using VIF"""
        X = self.df.drop('Loan_Status', axis=1)
        
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        
        print("\nVIF Scores:")
        print(vif_data)
        
        # Remove features with high VIF (>5)
        high_vif_features = vif_data[vif_data["VIF"] > 5]["Feature"]
        if not high_vif_features.empty:
            print(f"\nRemoving high VIF features: {list(high_vif_features)}")
            self.df.drop(high_vif_features, axis=1, inplace=True)
    
    def prepare_data(self):
        """Prepare data for modeling"""
        # Convert target to numerical
        self.df['Loan_Status'] = self.df['Loan_Status'].map({'Y': 1, 'N': 0})
        
        X = self.df.drop('Loan_Status', axis=1)
        y = self.df['Loan_Status']
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Step 6: Train and evaluate models"""
        model_results = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Store results
            model_results[name] = {
                'model': model,
                'cv_mean_score': np.mean(cv_scores),
             ''   'cv_std': np.std(cv_scores)
            }
            
            print(f"\n{name} - CV Mean Accuracy: {np.mean(cv_scores):.2%} (±{np.std(cv_scores):.2%})")
        
        return model_results
    
    def evaluate_models(self, model_results, X_test, y_test):
        """Evaluate models on test set"""
        evaluation_results = {}
        
        for name, result in model_results.items():
            model = result['model']
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report
            }
            
            print(f"\n{name} Evaluation:")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"ROC AUC Score: {roc_auc:.2f}")
            print("Confusion Matrix:")
            print(conf_matrix)
            print("\nClassification Report:")
            print(class_report)
            
            # Plot ROC curve
            self._plot_roc_curve(model, X_test, y_test, name)
        
        return evaluation_results
    
    def _plot_roc_curve(self, model, X_test, y_test, model_name):
        """Plot ROC curve for a model"""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
    
    def select_best_model(self, evaluation_results):
        """Select the best model based on evaluation"""
        best_model_name = max(
            evaluation_results.items(),
            key=lambda x: (x[1]['accuracy'], x[1]['roc_auc'])
        )[0]
        
        self.best_model = self.models[best_model_name]
        print(f"\nSelected best model: {best_model_name}")
        
        return self.best_model
    
    def save_model(self, filename='best_loan_model.pkl'):
        """Save the best model to a file"""
        if self.best_model:
            with open(filename, 'wb') as file:
                pickle.dump(self.best_model, file)
            print(f"Model saved as {filename}")
        else:
            print("No model to save. Train a model first.")

if __name__ == "__main__":
    # Initialize the system
    loan_system = LoanPredictionSystem()
    
    # Step 1-2: Load and clean data
    df = loan_system.load_data('loan_data.csv')
    
    # Step 3: Data visualization
    loan_system.visualize_data()
    
    # Step 4: Remove outliers
    loan_system.remove_outliers()
    
    # Step 5: Check multicollinearity
    loan_system.check_multicollinearity()
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = loan_system.prepare_data()
    
    # Step 6: Train models
    model_results = loan_system.train_models(X_train, y_train)
    
    # Evaluate models
    evaluation_results = loan_system.evaluate_models(model_results, X_test, y_test)
    
    # Select best model
    best_model = loan_system.select_best_model(evaluation_results)
    
    # Save the best model
    loan_system.save_model()