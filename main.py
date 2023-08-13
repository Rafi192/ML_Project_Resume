import os
import csv
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

# Loading the trained SVM model
model_path = 'D:\Resume_classification\Resume\Project\model_svm.pkl'
model = joblib.load(model_path)

def categorize_resume(resume_text):
    predicted_category = model.predict([resume_text])
    return predicted_category

def main():
    parser = argparse.ArgumentParser(description='Categorize resumes')
    parser.add_argument('D:\Resume_classification\Resume\Project\APPAREL', type=str, help='Path to directory containing resumes')
    args = parser.parse_args()

    input_dir = args.input_dir

    # Creating a CSV file to store categorized resumes
    csv_filename = 'categorized_resumes.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'category'])

        for filename in os.listdir(input_dir):
            if filename.endswith('.pdf'):
                with open(os.path.join(input_dir, filename), 'r') as file:
                    resume_text = file.read()
                    predicted_category = categorize_resume(resume_text)

                    # Moving the resume to the category folder
                    category_folder = os.path.join(input_dir, str(predicted_category))
                    os.makedirs(category_folder, exist_ok=True)
                    os.rename(os.path.join(input_dir, filename), os.path.join(category_folder, filename))

                    # Writing to CSV
                    csvwriter.writerow([filename, predicted_category])

if __name__ == '__main__':
    main()
