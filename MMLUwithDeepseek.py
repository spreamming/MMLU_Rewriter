import os
import logging
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    filename='mmlu_selection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    error_msg = "API Key not found. Please check your .env file."
    logging.error(error_msg)
    print(error_msg)
    exit(1)

print("API Key loaded successfully")
logging.info("API Key loaded successfully")

print("This script selects 20 questions from the MMLU dataset")
print("To process these questions with the DeepSeek model, run process_all_questions.py after this completes")
print("Note: The free tier of Together.AI allows only ~1.8 requests per minute for the DeepSeek model")

# Load MMLU dataset
try:
    print("Loading MMLU dataset...")
    logging.info("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    df = dataset.to_pandas()
    print(f"Dataset loaded: {len(df)} questions")
    logging.info(f"Dataset loaded: {len(df)} questions")
except Exception as e:
    error_msg = f"Error loading dataset: {e}"
    logging.error(error_msg)
    print(error_msg)
    exit(1)

# Select 20 random questions across subjects
subjects = df['subject'].unique()
samples_per_subject = max(1, 20 // len(subjects))
selected_questions = []

print(f"Found {len(subjects)} subjects")
print(f"Sampling {samples_per_subject} questions per subject")
logging.info(f"Found {len(subjects)} subjects")
logging.info(f"Sampling {samples_per_subject} questions per subject")

for subject in subjects:
    subject_df = df[df['subject'] == subject]
    sampled = subject_df.sample(n=min(samples_per_subject, len(subject_df)), random_state=42)
    selected_questions.append(sampled)

selected_df = pd.concat(selected_questions).sample(n=20, random_state=42).reset_index(drop=True)
print(f"Selected {len(selected_df)} questions for processing")
logging.info(f"Selected {len(selected_df)} questions for processing")

# Save selected questions as CSV for reference
selected_df.to_csv("selected_questions.csv", index=False)
print("Saved selected questions to selected_questions.csv")
logging.info("Saved selected questions to selected_questions.csv")

print("\nSample of selected questions:")
for i, row in selected_df.head(3).iterrows():
    print(f"\nQuestion {i+1}:")
    print(f"Subject: {row['subject']}")
    print(f"Question: {row['question']}")
    print(f"Choices: {row['choices']}")
    print(f"Answer Index: {row['answer']}")

print("\nTo process these questions and generate rewritten versions, run:")
print("python process_all_questions.py")
print("\nNote: This will take time due to rate limits for the free tier on Together.ai")
print("The script will handle rate limiting and save progress after each question")