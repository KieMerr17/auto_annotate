import openai
from env import openai_API_key
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from driving_errors import driving_errors, labels
from categories import categories

# Create a DataFrame with driving error examples and their corresponding categories
data = pd.DataFrame({
    'transcript': driving_errors,
    'category': labels
})

# Split the data into features (transcripts) and labels (categories)
X = data['transcript']
y = data['category']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))  # Unigrams and bigrams
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a classifier (Random Forest in this case)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Function to save the vectorizer
def save_vectorizer(vectorizer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vectorizer, f)

# Function to load the trained model
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Function to load the vectorizer
def load_vectorizer(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Function to save user corrections to a CSV file
def save_corrections(transcripts, categories, filename="corrections.csv"):
    corrections_df = pd.DataFrame({
        'transcript': transcripts,
        'category': categories
    })
    corrections_df.to_csv(filename, index=False)

# Function to load user corrections from a CSV file
def load_corrections(filename="corrections.csv"):
    try:
        corrections_df = pd.read_csv(filename)
        return corrections_df['transcript'].tolist(), corrections_df['category'].tolist()
    except FileNotFoundError:
        return [], []

# Function to save accuracy history to a CSV file
def save_accuracy_history(data, filename="accuracy_history.csv"):
    accuracy_df = pd.DataFrame(data)
    accuracy_df.to_csv(filename, index=False)

# Function to load accuracy history from a CSV file
def load_accuracy_history(filename="accuracy_history.csv"):
    try:
        accuracy_df = pd.read_csv(filename)
        return accuracy_df.to_dict(orient='records')
    except FileNotFoundError:
        return []

# Function to generate a new driving error using GPT
def generate_random_driving_error():
    openai.api_key = openai_API_key
    prompt = """
    Generate a random very short transcript of a driver explaining a driving error.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate a driving error that requires a driver to intervene."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.7
    )

    # Extract the driving error from the response
    driving_error = response['choices'][0]['message']['content'].strip()
    return driving_error

# Function to categorize a new transcript
def categorize_transcript_ml(transcript, vectorizer, model):
    # Preprocess and vectorize the transcript
    transcript_tfidf = vectorizer.transform([transcript])
    
    # Predict the category
    predicted_category = model.predict(transcript_tfidf)[0]
    
    return predicted_category

# Global variable to hold new transcripts and categories for retraining
new_transcripts = []
new_categories = []

# Function to validate the prediction and offer correction
def validate_prediction(transcript, predicted_category):
    print("\n---- Category Assessment ----\n")
    print("Transcript:", transcript)
    print("Predicted Category:", predicted_category)
    
    # Ask the user if the predicted category is correct
    correct = input("Is the prediction correct? (y/n): ").strip().lower()
    
    if correct == 'y':
        print("Great! The prediction is correct.")
        return predicted_category, 100.0  # 100% accuracy
    else:
        print("\nPlease select the correct category from the list below:")
        for idx, category in enumerate(categories, 1):
            print(f"{idx}. {category}")
        
        # Let the user choose the correct category by number
        correct_category_idx = int(input("Enter the number of the correct category (1-9): "))
        correct_category = categories[correct_category_idx - 1]
        
        print(f"Thank you! The correct category is: {correct_category}")
        
        # Store the corrected transcript and category for retraining
        new_transcripts.append(transcript)
        new_categories.append(correct_category)
        
        return correct_category, 0.0  # 0% accuracy if the prediction was incorrect

# Function to retrain the model with new data
def retrain_model(vectorizer, classifier, new_transcripts, new_categories):
    if len(new_transcripts) > 0:
        # Append the new corrected data to the original dataset
        additional_data = pd.DataFrame({
            'transcript': new_transcripts,
            'category': new_categories
        })

        global data
        data = pd.concat([data, additional_data], ignore_index=True)

        # Re-vectorize the full dataset
        X_full = data['transcript']
        y_full = data['category']
        X_full_tfidf = vectorizer.fit_transform(X_full)

        # Retrain the classifier
        classifier.fit(X_full_tfidf, y_full)

        print("\nModel retrained with user-corrected data.")
        
        # Save the updated model, vectorizer, and corrections
        save_model(classifier, "model.pkl")
        save_vectorizer(vectorizer, "vectorizer.pkl")
        save_corrections(new_transcripts, new_categories)

# Load model and vectorizer if they exist
try:
    classifier = load_model("model.pkl")
    vectorizer = load_vectorizer("vectorizer.pkl")
    print("Loaded existing model and vectorizer.")
except (FileNotFoundError, EOFError):
    print("No existing model or vectorizer found. Starting with a new model.")

# Load corrections if they exist
new_transcripts, new_categories = load_corrections()

# Load accuracy history if it exists
accuracy_history = load_accuracy_history()

# Loop to run the model and ask the user continuously
def run_model_loop(vectorizer, classifier):
    while True:
        # Generate a random driving error using GPT instead of selecting from a fixed list
        random_transcript = generate_random_driving_error()
        print("\nGenerated Driving Error:", random_transcript)
        
        # Predict the category for the generated error
        predicted_category = categorize_transcript_ml(random_transcript, vectorizer, classifier)
        final_category, accuracy = validate_prediction(random_transcript, predicted_category)
        
        print("\nFinal Category Used:", final_category)

        # Store the accuracy in history
        accuracy_history.append({"transcript": random_transcript, "predicted_category": predicted_category, "accuracy": accuracy})

        # Save the accuracy history after each prediction
        save_accuracy_history(accuracy_history)

        # Retrain the model after each user correction
        retrain_model(vectorizer, classifier, new_transcripts, new_categories)

        # Ask if the user wants to try another transcript
        continue_running = input("\nDo you want to categorize another driving error? (y/n): ").strip().lower()
        if continue_running != 'y':
            print("Exiting program. Goodbye!")
            break

# Run the model loop
run_model_loop(vectorizer, classifier)
