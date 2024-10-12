import pandas as pd
import openai
from env import OPENAI_API_KEY

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Define categories and category details
categories = {
    "Failed to Accelerate": "The car is stopped and fails to start when it should; it doesn't move away from stationary. This can occur at traffic lights, when merging into traffic, or when attempting to start from a stop sign. Consequences may include blocking traffic or causing unsafe situations behind the vehicle.",
    "Failed to Slow": "The car does not slow down when required (e.g., for traffic lights, other vehicles, pedestrians, or generally going too fast for the situation). This could result in near misses with pedestrians crossing the street or rear-end collisions in stop-and-go traffic.",
    "Failed to Remain Stopped": "The car starts moving when it should stay stationary, such as at a stop sign or red light. This may include situations where the vehicle incorrectly attempts to move off before it is safe, potentially endangering pedestrians or oncoming traffic.",
    "Failed to Maintain Speed": "The car fails to maintain a consistent speed or does not increase its speed when required. This may happen when navigating hills or during acceleration after a stop, leading to dangerous conditions on highways or merging lanes.",
    "Failed to Follow Route": "The car fails to follow the intended route, makes incorrect lane changes, or uses wrong indicators. This can result in unexpected maneuvers that may confuse other drivers or lead to missed exits.",
    "Failed to Overtake": "The car fails to go around an obstruction, vehicle, cyclist, or other things such as a double-parked car or bus. This can create dangerous situations, especially in traffic where a timely overtake is critical to avoid collisions.",
    "Failed for Early Turn": "The car turns too early or too soon, cutting the corner. This may lead to encroaching on pedestrian walkways or intersecting traffic unexpectedly, posing risks to other road users.",
    "Failed for Late Turn": "The car turns wide, turns late, or straightens the steering wheel too late, potentially crossing into other lanes or creating hazards for vehicles behind or alongside.",
    "Failed for Lane Position": "The car drifts within its lane or fails to maintain proper lane position, which can lead to side-swipe accidents or confusion for other drivers about the vehicle's intentions.",
    "Uncommanded Disengagement": "The autonomous system disengages without any input from the operator, which may happen during critical driving situations, leading to loss of control and increased risk of accidents.",
    "End of Run": "A disengagement to mark the end of the testing session, which should occur safely and without abrupt actions. The end of run signal should be clear and should not confuse other road users.",
    "Failed to Initiate Maneuver": "The car fails to start a required maneuver (e.g., a required lane change). This may happen during merging or when a prompt for a turn is given, risking accidents if other drivers do not anticipate the car’s actions.",
    "Close Proximity": "Disengaging for safety reasons while stationary due to something coming within close proximity of the vehicle, such as another vehicle, cyclist, or pedestrian, which may indicate a lack of awareness of surroundings.",
    "Lens Obscured": "A camera or sensor is blurred, affecting the vehicle's perception. This can lead to misinterpretation of surroundings, potentially resulting in incorrect speed or direction adjustments.",
    "Unprompted Maneuver": "The car performs a maneuver without a clear prompt, which could confuse passengers or other road users. Examples include sudden lane changes or unexpected turns.",
    "Lane Change": "Issues related to failing to change lanes as required, which can create hazards when merging or navigating traffic. This may involve ignoring indicators or misjudging the speed of nearby vehicles.",
    "Emergency Stop": "The vehicle operator makes an emergency stop to avoid danger. This is critical for preventing collisions but can lead to sudden disruptions in traffic flow.",
    "Accidental AVSO Intervention": "Unintentional intervention by the safety operator, possibly due to a false alarm or misinterpretation of a situation. This can disrupt the flow of autonomous operation and may confuse other road users.",
    "Uncategorized": "Situations that don't fit into other categories, often requiring further analysis to determine the appropriate classification.",
    "Diversion": "The plotted route cannot be driven on, so the operator is required to divert from the route. This may involve unexpected turns or navigating unfamiliar roads, which could impact safety.",
    "Emergency Service": "The vehicle interacts with emergency services (e.g., yielding to an ambulance). This requires quick and decisive action to clear the way, ensuring that emergency responders can reach their destination promptly."
}


category_details = {
    "Failed to Accelerate": [
        "Empty Zebra Crossing", "Yellow Box", "Green Light", "Turning", "Give Way", "Move Off in Traffic",
        "After Negotiating Traffic on Narrow Road", "Keep Clear Markings", "Stop Junction", "Roundabout", 
        "Pedestrian/Jaywalker Finished", "Clear Road", "Four Way Stop", "Turning Right on Red"
    ],
    "Failed to Slow": [
        "Above Speed Limit", "Approaching Junction", "Amber Light", "Approaching Potholes", "Red Light", 
        "At a Give Way", "Oncoming Vehicles", "Behind a Refuse Truck Working at the Roadside", "Non Priority Vehicles",
        "Behind Vehicle Stationary in Traffic", "Cyclist", "Other Dynamic", "Jaywalking", "Stop Sign", 
        "Primary Stop Line", "Undertaking", "Exceed Driver Set Speed", "Keeping Clear Markings", 
        "Pedestrian on Zebra Crossing", "Match Speed of Traffic", "Pedestrian at Junction or Informal Crossing Point",
        "Negotiating Traffic on Narrow Road", "Behind a Bus at a Bus Stop", "Roadworks", "Stop Junction", 
        "Speed Bump", "Stopped Across Pedestrian Crossing", "Vehicle Merging into Lane", "Width Restriction", "Yellow Box"
    ],
    "Failed to Remain Stopped": [
        "Amber Light", "Junction", "Keep Clear Markings", "Negotiating Traffic on Narrow Roads", 
        "Pedestrian on Zebra Crossing", "Red Light", "Stationary Traffic", "Waiting to Turn Across Traffic", 
        "Yellow Box", "Pedestrian at Junction or Informal Crossing Point", "Jaywalking", "Oncoming Vehicles", 
        "At a Give Way", "Stop Junction", "Non Priority Vehicles", "Cyclist", "Primary Stop Line", 
        "Rollback", "Four Way Stop", "Red Arrow"
    ],
    "Failed to Maintain Speed": [
        "Empty Zebra Crossing", "Green Light", "Match Traffic Speed", "Slowed Too Early", "Stay at Speed Limit", 
        "Turning", "Exceeding Speed Limit", "Variable Speed Limit"
    ],
    "Failed to Follow Route": [
        "Failed to Turn into Plotted Turn", "Incorrect Indicator", "Incorrect Lane for Continue Ahead", 
        "Incorrect Lane for Upcoming Turn", "Took Unplotted Turn"
    ],
    "Failed to Overtake": [
        "Incorrectly Initiated Lorry", "Failed to Initiate Lorry", "Failed to Complete Lorry", 
        "Incorrectly Initiated Bus", "Failed to Initiate Bus", "Failed to Complete Bus", 
        "Incorrectly Initiated Cyclist", "Failed to Initiate Cyclist", "Failed to Complete Cyclist", 
        "Incorrectly Initiated Double Parked Vehicle", "Failed to Initiate Double Parked Vehicle", "Failed to Complete Double Parked Vehicle", 
        "Incorrectly Initiated Line of Parked Vehicles", "Failed to Initiate Line of Parked Vehicles", "Failed to Complete Line of Parked Vehicles", 
        "Incorrectly Initiated Motorcycle", "Failed to Initiate Motorcycle", "Failed to Complete Motorcycle", 
        "Incorrectly Initiated Ongoing Vehicle Waiting to Turn", "Failed to Initiate Ongoing Vehicle Waiting to Turn", "Failed to Complete Ongoing Vehicle Waiting to Turn", 
        "Incorrectly Initiated Other (Dynamic)", "Failed to Initiate Other (Dynamic)", "Failed to Complete Other (Dynamic)", 
        "Incorrectly Initiated Other (Static)", "Failed to Initiate Other (Static)", "Failed to Complete Other (Static)", 
        "Incorrectly Initiated Single Parked Vehicle", "Failed to Initiate Single Parked Vehicle", "Failed to Complete Single Parked Vehicle"
    ],
    "Failed for Early Turn": [
        "Towards Cyclist", "Towards Motorcyclist", "Towards Kerb", "Towards Oncoming Lane", 
        "Towards Ongoing Lane", "Towards Ongoing Restricted Bus Lane", "Towards Ongoing Restricted Cycle Lane", 
        "Towards Other (Dynamic)", "Towards Other (Static)", "Towards Parked Vehicle", "Towards Pedestrian", 
        "Towards Roundabout"
    ],
    "Failed for Late Turn": [
        "Towards Cyclist", "Towards Motorcyclist", "Towards Kerb", "Towards Oncoming Lane", 
        "Towards Ongoing Lane", "Towards Ongoing Restricted Bus Lane", "Towards Ongoing Restricted Cycle Lane", 
        "Towards Other (Dynamic)", "Towards Other (Static)", "Towards Parked Vehicle", "Towards Pedestrian", 
        "Towards Roundabout"
    ],
    "Uncommanded Disengagement": [
        "Uncommanded Disengagement"
    ],
    "End of Run": [
        "End of Run"
    ],
    "Failed to Initiate Maneuver": [
        "Failed to Initiate Maneuver"
    ],
    "Close Proximity": [
        "Close Proximity"
    ],
    "Lens Obscured": [
        "Lens Obscured"
    ],
    "Unprompted Maneuver": [
        "Unprompted Maneuver"
    ],
    "Lane Change": [
        "Lane Change"
    ],
    "Emergency Stop": [
        "Emergency Stop"
    ],
    "Accidental AVSO Intervention": [
        "Accidental AVSO Intervention"
    ],
    "Uncategorized": [
        "Uncategorized"
    ],
    "Diversion": [
        "Diversion"
    ],
    "Emergency Service": [
        "Emergency Service"
    ]
}

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

# Function to interact with ChatGPT for category and subcategory prediction
def categorize_transcript_gpt(transcript):
    # Create a formatted string of categories and their subcategories
    category_prompt = "\n".join(
        [f"{cat}: {', '.join(subcategories)}" for cat, subcategories in category_details.items()]
    )

    # Construct the prompt including the list of categories and subcategories
    prompt = (
        f"Categorize the following transcript into one of the provided categories and subcategories, "
        f"based on the explanations given in the category descriptions here: {categories}.\n\n"
        f"Transcript: {transcript}\n\n"
        f"Categories and subcategories:\n{category_prompt}\n\n"
        "Respond in this format:\nCategory: <category>\nSubcategory: <subcategory>\n"
        "Ensure the subcategory is valid for the selected category."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that categorizes driving transcripts."},
            {"role": "user", "content": prompt},
        ]
    )

    # Extract the category and subcategory from the response
    output = response['choices'][0]['message']['content']
    return output

# Main function to run the categorization
def main():
    transcripts = load_transcripts()
    results = []

    for transcript in transcripts:
        category_info = categorize_transcript_gpt(transcript)
        results.append((transcript, category_info))

    # Save results to a CSV file
    df = pd.DataFrame(results, columns=["Transcript", "Categorization"])
    df.to_csv("categorized_transcripts.csv", index=False)

# Execute the main function
if __name__ == "__main__":
    main()
