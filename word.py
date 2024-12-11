import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

# Load the dataset from the CSV file
df = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Text Classification\bbc-text.csv')

# Display the first few rows of the dataframe
print(df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)

# Create a pipeline that combines TF-IDF vectorization and the classifier
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english'),  # Use TF-IDF and remove stop words
    MultinomialNB()  # Naive Bayes classifier
)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))