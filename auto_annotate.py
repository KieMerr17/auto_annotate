import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import logging
from categories import categories

# Function to load existing data from CSV and handle potential parsing errors
def load_existing_data(filename="data.csv"):
    try:
        existing_data = pd.read_csv(filename, on_bad_lines='skip')
        return existing_data
    except FileNotFoundError:
        print("No existing data file found, starting fresh.")
        return pd.DataFrame(columns=['transcript', 'category', 'subcategory'])  # Add subcategory column
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file: {e}")
        return pd.DataFrame(columns=['transcript', 'category', 'subcategory'])  # Add subcategory column

# Load existing data
data = load_existing_data("data.csv")

# Split the data into features and labels (categories and subcategories)
X = data['transcript']
y_category = data['category']
y_subcategory = data['subcategory']
X_train, X_test, y_category_train, y_category_test, y_subcategory_train, y_subcategory_test = train_test_split(
    X, y_category, y_subcategory, test_size=0.2, random_state=42
)

# Convert the text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train two classifiers: one for category and one for subcategory
category_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
subcategory_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

category_classifier.fit(X_train_tfidf, y_category_train)
subcategory_classifier.fit(X_train_tfidf, y_subcategory_train)

# Predict on the test set
y_category_pred = category_classifier.predict(X_test_tfidf)
y_subcategory_pred = subcategory_classifier.predict(X_test_tfidf)

category_accuracy = accuracy_score(y_category_test, y_category_pred)
subcategory_accuracy = accuracy_score(y_subcategory_test, y_subcategory_pred)
logging.info(f'Initial model accuracy - Category: {category_accuracy:.2f}%, Subcategory: {subcategory_accuracy:.2f}%')

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

def categorize_transcript_ml(transcript, vectorizer, category_model, subcategory_model):
    transcript_tfidf = vectorizer.transform([transcript])
    predicted_category = category_model.predict(transcript_tfidf)[0]
    predicted_subcategory = subcategory_model.predict(transcript_tfidf)[0]
    return predicted_category, predicted_subcategory

# Load model and vectorizer, if available
try:
    category_classifier = load_model("category_model.pkl")
    subcategory_classifier = load_model("subcategory_model.pkl")
    vectorizer = load_vectorizer("vectorizer.pkl")
    print("Loaded existing models and vectorizer.")
except (FileNotFoundError, EOFError):
    print("No existing models or vectorizer found. Starting with new models.")

accuracy_history = load_accuracy_history()

def calculate_overall_accuracy(accuracy_history):
    if len(accuracy_history) == 0:
        return 0.0
    total_accuracy = sum(entry['accuracy'] for entry in accuracy_history)
    overall_accuracy = total_accuracy / len(accuracy_history)
    return overall_accuracy

def display_subcategories(category):
    if isinstance(categories[category], dict):
        subcategories = categories[category]
        print(f"\nSubcategories under '{category}':")
        for idx, subcat in enumerate(subcategories.keys(), 1):
            print(f"{idx}. {subcat}")
        return True
    return False

def validate_prediction(transcript, predicted_category, predicted_subcategory):
    print("\n- - - Category and Subcategory Assessment - - -\n")
    print("Transcript:\n", transcript, "\n")
    print("Predicted Category:\n", predicted_category, "\n")
    print("Predicted Subcategory:\n", predicted_subcategory)

    # Ask the user if the predictions are correct
    while True:
        print("")
        correct = input("Is the prediction correct? (y/n): ").strip().lower()
        if correct in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")
    
    if correct == 'y':
        print("\nGreat! The prediction is correct.")
        return predicted_category, predicted_subcategory, 100.0  # 100% accuracy
    else:
        print("\nPlease select the correct category from the list below:")
        for idx, category in enumerate(categories.keys(), 1):
            print(f"{idx}. {category}")
        
        # Let the user choose the correct category by number
        while True:
            try:
                correct_category_idx = int(input("Enter the number of the correct category: "))
                if 1 <= correct_category_idx <= len(categories):
                    correct_category = list(categories.keys())[correct_category_idx - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(categories)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        # Prepare to select the subcategory
        correct_subcategory = None  # Initialize subcategory

        # Check if the selected category has subcategories
        subcategories = categories[correct_category]

        if isinstance(subcategories, list) and subcategories:  # Simple list of subcategories
            print(f"\nSubcategories under '{correct_category}':")
            for idx, subcat in enumerate(subcategories, 1):
                print(f"{idx}. {subcat}")
            
            # Let the user choose the correct subcategory if applicable
            while True:
                try:
                    correct_subcategory_idx = int(input("Enter the number of the correct subcategory: "))
                    if 1 <= correct_subcategory_idx <= len(subcategories):
                        correct_subcategory = subcategories[correct_subcategory_idx - 1]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(subcategories)}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
        
        elif isinstance(subcategories, dict) and subcategories:  # Check for nested dictionary of subcategories
            for subcat_type, subcat_list in subcategories.items():
                print(f"\nSubcategories under '{correct_category}' ({subcat_type}):")
                for idx, subcat in enumerate(subcat_list, 1):
                    print(f"{idx}. {subcat}")
            
            # Let the user choose the correct subcategory type first
            while True:
                try:
                    correct_subcategory_type_idx = int(input("Select the subcategory type number: "))
                    if 1 <= correct_subcategory_type_idx <= len(subcategories):
                        correct_subcategory_type = list(subcategories.keys())[correct_subcategory_type_idx - 1]
                        correct_subcategory_list = subcategories[correct_subcategory_type]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(subcategories)}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

            # Now select the actual subcategory from the chosen type
            while True:
                print(f"\nSelect the correct subcategory from '{correct_subcategory_type}':")
                for idx, subcat in enumerate(correct_subcategory_list, 1):
                    print(f"{idx}. {subcat}")
                
                try:
                    correct_subcategory_idx = int(input("Enter the number of the correct subcategory: "))
                    if 1 <= correct_subcategory_idx <= len(correct_subcategory_list):
                        correct_subcategory = correct_subcategory_list[correct_subcategory_idx - 1]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(correct_subcategory_list)}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
        
        # If no subcategory was selected, provide feedback accordingly
        if correct_subcategory is None:
            print("")
            print(f"Thank you! \nThe correct category is: \n\n- {correct_category}. (No subcategory needed.)")
            return correct_category, None, 0.0  # Return None for subcategory if not applicable
        else:
            print("")
            print(f"Thank you! \n\nThe correct category and subcategory are: \n\n- {correct_category}, \n- {correct_subcategory}")
            return correct_category, correct_subcategory, 0.0  # 0% accuracy if the prediction was incorrect

def self_learn(transcript, final_category, final_subcategory):
    # Load existing data
    existing_data = load_existing_data("data.csv")

    # Create a new DataFrame for the new entry
    additional_data = pd.DataFrame({'transcript': [transcript], 'category': [final_category], 'subcategory': [final_subcategory]})

    # Append the new data to the existing data
    combined_data = pd.concat([existing_data, additional_data], ignore_index=True)

    # Save the combined data back to the CSV file
    combined_data.to_csv("data.csv", index=False)

    # Re-vectorize the full dataset
    X_full = combined_data['transcript']
    y_full_category = combined_data['category']
    y_full_subcategory = combined_data['subcategory']
    X_full_tfidf = vectorizer.fit_transform(X_full)

    # Retrain the classifiers
    category_classifier.fit(X_full_tfidf, y_full_category)
    subcategory_classifier.fit(X_full_tfidf, y_full_subcategory)

    # Save the updated models and vectorizer
    save_model(category_classifier, "category_model.pkl")
    save_model(subcategory_classifier, "subcategory_model.pkl")
    save_vectorizer(vectorizer, "vectorizer.pkl")

    print("\nTraining Model with the new data...")

def run_model_loop(vectorizer, category_classifier, subcategory_classifier):
    while True:
        print("\n- - - New Driving Error - - -\n")
        user_transcript = input("Please describe the driving error: ").strip()

        predicted_category, predicted_subcategory = categorize_transcript_ml(user_transcript, vectorizer, category_classifier, subcategory_classifier)
        final_category, final_subcategory, accuracy = validate_prediction(user_transcript, predicted_category, predicted_subcategory)

        # print("\nFinal Category Used:", final_category)
        # print("Final Subcategory Used:", final_subcategory)

        # Auto-retrain if the prediction is correct
        if accuracy == 100.0:
            self_learn(user_transcript, final_category, final_subcategory)
        else:
            # Ask whether to retrain the model after an incorrect prediction
            while True:
                print("")
                retrain = input("Do you want to retrain the model with this correction? (y/n): ").strip().lower()
                if retrain in ['y', 'n']:
                    break
                else:
                    print("Invalid input. Please enter 'y' for yes or 'n' for no.")
            
            if retrain == 'y':
                self_learn(user_transcript, final_category, final_subcategory)

        accuracy_history.append({"transcript": user_transcript, "predicted_category": predicted_category, "predicted_subcategory": predicted_subcategory, "accuracy": accuracy})
        save_accuracy_history(accuracy_history)

        overall_accuracy = calculate_overall_accuracy(accuracy_history)
        print(f"\nCurrent Overall Accuracy: {overall_accuracy:.2f}% \n")

# Run the model loop
run_model_loop(vectorizer, category_classifier, subcategory_classifier)
