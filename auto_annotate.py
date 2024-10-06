import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import logging
from categories import categories

# Configure logging
logging.basicConfig(filename='model.log', level=logging.INFO)

# Function to load existing data from CSV and handle potential parsing errors
def load_existing_data(filename="data.csv"):
    try:
        existing_data = pd.read_csv(filename, on_bad_lines='skip')  
        return existing_data
    except FileNotFoundError:
        print("No existing data file found, starting fresh.")
        return pd.DataFrame(columns=['transcript', 'category'])  # Return empty DataFrame with columns
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file: {e}")
        return pd.DataFrame(columns=['transcript', 'category'])  # Return empty DataFrame with columns

# Load existing data
data = load_existing_data("data.csv")

# Split the data into features and labels
X = data['transcript']
y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a classifier (Random Forest)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f'Initial model accuracy: {accuracy:.2f}%')

# Functions for saving and loading models and vectorizers
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def save_vectorizer(vectorizer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_vectorizer(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_accuracy_history(data, filename="accuracy_history.csv"):
    accuracy_df = pd.DataFrame(data)
    accuracy_df.to_csv(filename, index=False)

def load_accuracy_history(filename="accuracy_history.csv"):
    try:
        accuracy_df = pd.read_csv(filename)
        return accuracy_df.to_dict(orient='records')
    except FileNotFoundError:
        return []

def categorize_transcript_ml(transcript, vectorizer, model):
    transcript_tfidf = vectorizer.transform([transcript])
    predicted_category = model.predict(transcript_tfidf)[0]
    return predicted_category

# Load model and vectorizer, if available
try:
    classifier = load_model("model.pkl")
    vectorizer = load_vectorizer("vectorizer.pkl")
    print("Loaded existing model and vectorizer.")
except (FileNotFoundError, EOFError):
    print("No existing model or vectorizer found. Starting with a new model.")

accuracy_history = load_accuracy_history()

def calculate_overall_accuracy(accuracy_history):
    if len(accuracy_history) == 0:
        return 0.0
    total_accuracy = sum(entry['accuracy'] for entry in accuracy_history)
    overall_accuracy = total_accuracy / len(accuracy_history)
    return overall_accuracy

def validate_prediction(transcript, predicted_category):
    print("\n- - - Category Assessment - - -\n")
    print("Transcript:\n", transcript)
    print("\nPredicted Category:\n", predicted_category)
    
    # Ask the user if the predicted category is correct
    while True:
        correct = input("Is the prediction correct? (y/n): ").strip().lower()
        if correct in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")
    
    if correct == 'y':
        print("Great! The prediction is correct.")
        return predicted_category, 100.0  # 100% accuracy
    else:
        print("\nPlease select the correct category from the list below:")
        for idx, category in enumerate(categories, 1):
            print(f"{idx}. {category}")
        
        # Let the user choose the correct category by number
        while True:
            try:
                correct_category_idx = int(input("Enter the number of the correct annotation: "))
                if 1 <= correct_category_idx <= len(categories):
                    correct_category = categories[correct_category_idx - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(categories)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
        print(f"Thank you! The correct category is: {correct_category}")
        return correct_category, 0.0  # 0% accuracy if the prediction was incorrect

def self_learn(transcript, final_category):
    # Load existing data
    existing_data = load_existing_data("data.csv")

    # Create a new DataFrame for the new entry
    additional_data = pd.DataFrame({'transcript': [transcript], 'category': [final_category]})

    # Append the new data to the existing data
    combined_data = pd.concat([existing_data, additional_data], ignore_index=True)

    # Save the combined data back to the CSV file
    combined_data.to_csv("data.csv", index=False)

    # Re-vectorize the full dataset
    X_full = combined_data['transcript']
    y_full = combined_data['category']
    X_full_tfidf = vectorizer.fit_transform(X_full)

    # Retrain the classifier
    classifier.fit(X_full_tfidf, y_full)

    # Save the updated model and vectorizer
    save_model(classifier, "model.pkl")
    save_vectorizer(vectorizer, "vectorizer.pkl")

    print("\nTraining Model with the new data...")

def run_model_loop(vectorizer, classifier):
    while True:
        print("\n- - - New Driving Error - - -\n")
        user_transcript = input("Please describe the driving error: ").strip()

        predicted_category = categorize_transcript_ml(user_transcript, vectorizer, classifier)
        final_category, accuracy = validate_prediction(user_transcript, predicted_category)

        print("\nFinal Category Used:", final_category)

        # Auto-retrain if the prediction is correct
        if accuracy == 100.0:
            self_learn(user_transcript, final_category)
        else:
            # Ask whether to retrain the model after an incorrect prediction
            while True:
                retrain = input("Do you want to retrain the model with this correction? (y/n): ").strip().lower()
                if retrain in ['y', 'n']:
                    break
                else:
                    print("Invalid input. Please enter 'y' for yes or 'n' for no.")
            
            if retrain == 'y':
                self_learn(user_transcript, final_category)

        accuracy_history.append({"transcript": user_transcript, "predicted_category": predicted_category, "accuracy": accuracy})
        save_accuracy_history(accuracy_history)

        overall_accuracy = calculate_overall_accuracy(accuracy_history)
        print(f"\nCurrent Overall Accuracy: {overall_accuracy:.2f}% \n")

# Run the model loop
run_model_loop(vectorizer, classifier)
