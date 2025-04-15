import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering
for df in [train, test]:
    # Extract title from name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+).', expand=False)
    # Consolidate rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                       'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Create family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Fill missing values
    df['Embarked'] = df['Embarked'].fillna('S')
    df['AgeBin'] = pd.qcut(df['Age'], 8, labels=False, duplicates='drop')
    df['FarePerPerson'] = df['Fare'] / (df['FamilySize'] + 1e-6)
    df['CabinDeck'] = df['Cabin'].str[0].fillna('Unknown')

# Define features
numerical_cols = ['AgeBin', 'FarePerPerson', 'SibSp', 'Parch']
categorical_cols = ['Pclass', 'Sex',
                    'Embarked', 'Title', 'IsAlone']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', SelectFromModel(RandomForestClassifier(n_estimators=100))),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Prepare data
X = train[numerical_cols + categorical_cols]
y = train['Survived']
test_data = test[numerical_cols + categorical_cols]

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Hyperparameter Grid
param_grid = {
    'classifier__n_estimators': [200, 300],
    'classifier__max_depth': [5, 7],
    'classifier__min_samples_split': [2, 5],
    'classifier__class_weight': [None, 'balanced'],
    'feature_selector__max_features': [15, 20]
}

# Cross-Validated Search
cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(model, param_grid, cv=cv,
                           scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")

# Final Submission
best_model = grid_search.best_estimator_
cv_val_preds = best_model.predict(X_val)
# Score
print(f"CV Val Accuracy: {accuracy_score(y_val, cv_val_preds):.4f}")

# Predict on test data
test_preds = best_model.predict(test_data)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")
