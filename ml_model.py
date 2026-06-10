import pymysql
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Database connect
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    database="career_db"
)

# Data load
query = "SELECT * FROM students"
df = pd.read_sql(query, conn)

print("Data loaded! Total rows:", len(df))
print(df.head())

# Text to numbers
le_subject = LabelEncoder()
le_interest = LabelEncoder()
le_career = LabelEncoder()

df['subject'] = le_subject.fit_transform(df['subject'])
df['interest'] = le_interest.fit_transform(df['interest'])
df['predicted_career'] = le_career.fit_transform(df['predicted_career'])

# Input / Output
X = df[['marks1', 'marks2', 'subject', 'interest']]
y = df['predicted_career']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Test prediction
new_data = pd.DataFrame([[90, 85,
    le_subject.transform(['math'])[0],
    le_interest.transform(['ai'])[0]]],
    columns=['marks1', 'marks2', 'subject', 'interest'])

prediction = model.predict(new_data)
result = le_career.inverse_transform(prediction)
print("Predicted Career:", result[0])