import pymysql
import random

conn = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    database="career_db"
)
cursor = conn.cursor()

matric_groups = ['Science General', 'Computer Science', 'Biology', 'Arts', 'Commerce']
inter_groups = ['ICS', 'Pre-Engineering', 'Pre-Medical', 'Commerce', 'Arts']
subjects = ['math', 'biology', 'commerce', 'physics', 'computer', 'chemistry', 'english', 'urdu']
interests = ['programming', 'medical', 'business', 'engineering', 'ai', 'law', 'art', 'architecture', 'cybersecurity']
family_influences = ['Yes', 'No']
family_careers = ['doctor', 'engineer', 'businessman', 'teacher', 'lawyer', 'none']

careers = {
    ('math', 'programming'): 'software engineer',
    ('math', 'ai'): 'data scientist',
    ('math', 'cybersecurity'): 'cybersecurity expert',
    ('biology', 'medical'): 'doctor',
    ('commerce', 'business'): 'businessman',
    ('physics', 'engineering'): 'engineer',
    ('computer', 'programming'): 'software engineer',
    ('computer', 'ai'): 'data scientist',
    ('computer', 'cybersecurity'): 'cybersecurity expert',
    ('english', 'law'): 'lawyer',
    ('urdu', 'art'): 'artist',
    ('chemistry', 'medical'): 'doctor',
    ('physics', 'architecture'): 'architect',
}

# Delete old data and insert fresh
cursor.execute("DELETE FROM students")

for _ in range(500):
    m1 = random.randint(50, 100)
    m2 = random.randint(50, 100)
    sub = random.choice(subjects)
    interest = random.choice(interests)
    matric_grp = random.choice(matric_groups)
    inter_grp = random.choice(inter_groups)
    fam_inf = random.choice(family_influences)
    fam_career = random.choice(family_careers) if fam_inf == 'Yes' else 'none'
    career = careers.get((sub, interest), 'software engineer')

    cursor.execute(
        "INSERT INTO students (marks1, marks2, subject, interest, predicted_career, matric_group, inter_group, family_influence, family_career) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (m1, m2, sub, interest, career, matric_grp, inter_grp, fam_inf, fam_career)
    )

conn.commit()
print("500 rows inserted with all columns!")