import pandas as pd
import openai
from categories import categories
from env import OPENAI_API_KEY

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Function to load existing data from CSV and handle potential parsing errors
def load_existing_data(filename="data.csv"):
    try:
        existing_data = pd.read_csv(filename, on_bad_lines='skip')
        return existing_data
    except FileNotFoundError:
        print("No existing data file found, starting fresh.")
        return pd.DataFrame(columns=['transcript', 'category', 'subcategory'])
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file: {e}")
        return pd.DataFrame(columns=['transcript', 'category', 'subcategory'])

# Function to load new transcripts from a file
def load_transcripts(filename="transcripts.txt"):
    try:
        with open(filename, 'r') as file:
            transcripts = file.readlines()
        transcripts = [t.strip() for t in transcripts if t.strip()]  # Remove empty lines
        return transcripts
    except FileNotFoundError:
        print(f"Transcripts file '{filename}' not found.")
        return []

# Load existing accuracy history
def load_accuracy_history(filename="accuracy_history.csv"):
    try:
        accuracy_df = pd.read_csv(filename)
        return accuracy_df.to_dict(orient='records')
    except FileNotFoundError:
        return []

def save_accuracy_history(data, filename="accuracy_history.csv"):
    accuracy_df = pd.DataFrame(data)
    accuracy_df.to_csv(filename, index=False)

accuracy_history = load_accuracy_history()

# Function to calculate overall accuracy
def calculate_overall_accuracy(accuracy_history):
    if len(accuracy_history) == 0:
        return 0.0
    total_accuracy = sum(entry['accuracy'] for entry in accuracy_history)
    overall_accuracy = total_accuracy / len(accuracy_history)
    return overall_accuracy

# Function to interact with ChatGPT for category and subcategory prediction
def categorize_transcript_gpt(transcript):
    # Create a formatted string of categories and their subcategories
    category_prompt = "\n".join(
        [f"{category}: {', '.join(subcategories)}" for category, subcategories in categories.items()]
    )

    # Construct the prompt including the list of categories and subcategories
    prompt = (
        f"Given the following transcript, please categorize it into one of the provided categories and subcategories."
        f"\nTranscript: {transcript}\n\n"
        "Categories and their respective subcategories are:\n"
        f"{category_prompt}\n\n"
        "Please respond with the format:\nCategory: <category>\nSubcategory: <subcategory>"
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps classify transcripts into appropriate categories and subcategories from a predefined list."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5
    )

    gpt_response = response['choices'][0]['message']['content'].strip()
    lines = gpt_response.split('\n')
    
    category = lines[0].replace("Category:", "").strip()
    subcategory = lines[1].replace("Subcategory:", "").strip()

    return category, subcategory


# Function to validate prediction and allow manual correction
def validate_prediction(transcript, predicted_category, predicted_subcategory):
    print("\n- - - Category and Subcategory Assessment - - -")
    print("\nTranscript:\n", transcript)
    print("\nPredicted Category:\n", predicted_category)
    print("\nPredicted Subcategory:\n", predicted_subcategory)

    # Ask the user if the predictions are correct
    while True:
        correct = input("\nIs the prediction correct? (y/n): ").strip().lower()
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

        correct_subcategory = None  # Initialize subcategory
        subcategories = categories[correct_category]

        # Check if the selected category has subcategories
        if isinstance(subcategories, list) and subcategories:
            print(f"\nSubcategories under '{correct_category}':")
            for idx, subcat in enumerate(subcategories, 1):
                print(f"{idx}. {subcat}")
            
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
        
        print("")
        print(f"Thank you! \n\nThe correct category and subcategory are: \n\n- {correct_category}, \n- {correct_subcategory}")
        return correct_category, correct_subcategory, 0.0  # 0% accuracy if incorrect

# Function to run the model loop and process transcripts
def run_model_loop(transcript_file):
    transcripts = load_transcripts(transcript_file)

    if not transcripts:
        print("No transcripts found to process.")
        return

    for user_transcript in transcripts:
        # Use GPT to predict category and subcategory
        predicted_category, predicted_subcategory = categorize_transcript_gpt(user_transcript)
        final_category, final_subcategory, accuracy = validate_prediction(user_transcript, predicted_category, predicted_subcategory)

        accuracy_history.append({"transcript": user_transcript, "predicted_category": predicted_category, "predicted_subcategory": predicted_subcategory, "accuracy": accuracy})
        save_accuracy_history(accuracy_history)

        overall_accuracy = calculate_overall_accuracy(accuracy_history)
        print(f"\nCurrent Overall Accuracy: {overall_accuracy:.2f}% \n")

# Run the model loop with automatic transcript processing from a file
run_model_loop("transcripts.txt")
