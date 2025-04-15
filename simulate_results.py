import pandas as pd
import os

# Load selected questions
if not os.path.exists("selected_questions.csv"):
    print("Error: selected_questions.csv not found")
    exit(1)

selected_df = pd.read_csv("selected_questions.csv")
print(f"Loaded {len(selected_df)} selected questions")

# Load existing rewritten samples
rewritten_samples = []
if os.path.exists("mmlu_rewritten_samples.csv"):
    existing_df = pd.read_csv("mmlu_rewritten_samples.csv")
    print(f"Found {len(existing_df)} existing rewritten samples")
    
    # Convert to list of dictionaries
    if len(existing_df) > 0:
        for i, row in existing_df.iterrows():
            rewritten_samples.append(dict(row))

# Simulate remaining rewritten questions
for i, row in selected_df.iterrows():
    # Skip questions that are already rewritten
    if any(sample.get('original_index') == i for sample in rewritten_samples):
        continue
    
    # Simple simulation of a rewritten question
    question = row['question']
    choices_raw = row['choices']
    answer_idx = row['answer']
    
    # Parse choices
    import re
    choices = []
    if isinstance(choices_raw, str):
        matches = re.findall(r"'([^']*)'", choices_raw)
        if matches:
            choices = matches
        else:
            choices = choices_raw.split()
    else:
        choices = choices_raw
    
    # Get correct answer if possible
    correct_answer = "Unknown"
    if answer_idx < len(choices):
        correct_answer = choices[answer_idx]
    
    # Create simulated rewrite
    simulated = {
        "original_index": i,
        "subject": row['subject'],
        "original_question": question,
        "rewritten_question": f"[Simulated] Rewritten version of question about {row['subject']}",
        "choices": str(choices),
        "correct_answer": correct_answer
    }
    
    rewritten_samples.append(simulated)

# Save simulated results to CSV
simulated_df = pd.DataFrame(rewritten_samples)
simulated_df.to_csv("simulated_samples.csv", index=False)
print(f"Saved {len(simulated_df)} simulated samples to simulated_samples.csv")

# Display some examples
print("\nSample of simulated questions:")
for i, row in simulated_df.head(5).iterrows():
    print(f"\nSample {i+1}:")
    print(f"Subject: {row['subject']}")
    print(f"Original: {row['original_question'][:100]}..." if len(row['original_question']) > 100 else row['original_question'])
    print(f"Rewritten: {row['rewritten_question']}")
    print(f"Choices: {row['choices']}")
    print(f"Correct Answer: {row['correct_answer']}") 