import os
import pandas as pd
import re
import time
import logging
import json
from datetime import datetime
from together import Together
from dotenv import load_dotenv
import sys

# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fast_process.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    print("Starting fast processor...")
    logging.info("Starting fast processor")
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("API key not found in .env file")
        return
        
    print("API key loaded")
    
    # Initialize Together client
    try:
        client = Together(api_key=api_key)
        print("Together client initialized")
    except Exception as e:
        print(f"Error initializing Together client: {e}")
        return
        
    # Load question
    try:
        if not os.path.exists("selected_questions.csv"):
            print("selected_questions.csv not found")
            return
            
        df = pd.read_csv("selected_questions.csv")
        print(f"Loaded {len(df)} questions from CSV")
        
        # Process the first question
        question_idx = 0  # Process the first question
        row = df.iloc[question_idx]
        
        # Extract question info
        question_text = row['question']
        subject = row['subject']
        choices_raw = row['choices']
        answer_idx = int(row['answer'])
        
        print(f"\nProcessing question: {subject}")
        print(f"Question text: {question_text[:50]}...")
        
        # Parse choices
        choices = []
        try:
            if isinstance(choices_raw, str):
                # Extract quoted strings
                matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", choices_raw)
                if matches:
                    for match in matches:
                        choice = match[0] if match[0] else match[1]
                        choices.append(choice)
                else:
                    # Try splitting by newlines
                    parts = choices_raw.strip("[]").split("\\n")
                    if len(parts) > 1:
                        choices = [p.strip().strip("'\"") for p in parts if p.strip()]
                    else:
                        # Last resort: split by spaces
                        choices = [c.strip().strip("'\"") for c in choices_raw.strip("[]").split() if c.strip()]
        except Exception as e:
            print(f"Error parsing choices: {e}")
            choices = ["Choice A", "Choice B", "Choice C", "Choice D"]
            
        print(f"Parsed choices: {choices}")
        
        # Get answer
        answer = choices[answer_idx] if 0 <= answer_idx < len(choices) else "Unknown"
        print(f"Answer: {answer}")
        
        # Prepare prompt
        prompt = f"""
Please rewrite the following multiple-choice question to make it more challenging while preserving its essence and correct answer.

Original Question: {question_text}

Choices:
"""
        
        # Add choices
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
            
        prompt += f"\nCorrect Answer: {answer}\n\n"
        prompt += """
Rewrite the question in the following JSON format:
{
  "rewritten_question": "Your rewritten question here",
  "rewritten_choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
  "rewritten_answer": "The correct answer choice letter"
}
"""
        
        print("\nCalling API...")
        try:
            # Make API call directly
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                messages=[
                    {"role": "system", "content": "You are an expert in question design. Follow instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.5,
                timeout=60
            )
            
            response_text = response.choices[0].message.content.strip()
            print("\nAPI response received!")
            
            # Save raw response
            with open("fast_response_raw.txt", "w", encoding="utf-8") as f:
                f.write(response_text)
            print("Raw response saved to fast_response_raw.txt")
            
            # Try to parse JSON
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_content = json_match.group(1)
                data = json.loads(json_content)
                
                # Extract data
                rewritten_question = data.get('rewritten_question', '')
                rewritten_choices = data.get('rewritten_choices', [])
                rewritten_answer = data.get('rewritten_answer', '')
                
                # Create result
                result = {
                    "original_index": question_idx,
                    "subject": subject,
                    "original_question": question_text,
                    "rewritten_question": rewritten_question,
                    "choices": rewritten_choices,
                    "correct_answer": rewritten_answer
                }
                
                # Save to CSV
                output_file = "fast_processed.csv"
                pd.DataFrame([result]).to_csv(output_file, index=False)
                print(f"Result saved to {output_file}")
                
                # Print result
                print("\nRewritten Question:")
                print(rewritten_question)
                print("\nRewritten Choices:")
                for choice in rewritten_choices:
                    print(f"- {choice}")
                print(f"\nCorrect Answer: {rewritten_answer}")
                
            else:
                print("Could not find JSON content in response")
                print(f"Response text: {response_text}")
                
        except Exception as e:
            print(f"Error calling API: {e}")
        
    except Exception as e:
        print(f"Error processing: {e}")
        
    print("\nProcessing complete")

if __name__ == "__main__":
    main() 