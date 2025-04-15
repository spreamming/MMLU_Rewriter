import os
import pandas as pd
import re
import time
import logging
from datetime import datetime
import traceback
from together import Together
from dotenv import load_dotenv
import json

# Set up logging
logging.basicConfig(
    filename='batch_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global settings
API_TIMEOUT = 120  # API call timeout in seconds
MAX_RETRIES = 3    # Maximum retry attempts

def initialize_client():
    """Initialize the Together API client"""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please set TOGETHER_API_KEY in .env file.")
        
        # Initialize Together client
        together_client = Together(api_key=api_key)
        print("API key loaded successfully and Together client initialized.")
        logging.info("API key loaded successfully and Together client initialized.")
        
        return together_client
        
    except Exception as e:
        error_msg = f"Failed to initialize Together client: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return None

def call_api(together_client, prompt, retry_count=0):
    """
    Call the Together API without waiting time (for smaller models).
    
    Args:
        together_client: The Together client instance
        prompt: The prompt to send to the model
        retry_count: Current retry count
        
    Returns:
        The API response, or None if all retries failed
    """
    try:
        print(f"Calling API (attempt {retry_count + 1}/{MAX_RETRIES + 1})...")
        logging.info(f"Calling API (attempt {retry_count + 1}/{MAX_RETRIES + 1})...")
        
        # Make the API call
        response = together_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  # Using smaller DeepSeek model
            messages=[{
                "role": "system", 
                "content": "You are an expert in question design. Follow instructions exactly and provide only what is requested."
            }, {
                "role": "user", 
                "content": prompt
            }],
            max_tokens=500,
            temperature=0.5,  # Lower temperature for more consistent output
            timeout=API_TIMEOUT
        )
        return response
    except Exception as e:
        error_message = str(e)
        error_msg = f"Error calling API: {error_message}"
        logging.error(error_msg)
        print(error_msg)
        
        # Check if should retry
        if retry_count < MAX_RETRIES:
            next_retry = retry_count + 1
            retry_msg = f"Error occurred. Retrying immediately (attempt {next_retry + 1}/{MAX_RETRIES + 1})..."
            logging.warning(retry_msg)
            print(retry_msg)
            return call_api(together_client, prompt, next_retry)
        else:
            raise e  # Re-raise if max retries exceeded

def parse_choices(choices_raw):
    """Parse choices from raw string format"""
    choices = []
    try:
        # Try to parse the choices from raw format
        if isinstance(choices_raw, str):
            if "'" in choices_raw or '"' in choices_raw:
                # Extract quoted strings
                matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", choices_raw)
                if matches:
                    for match in matches:
                        # Each match is a tuple of groups, use the non-empty one
                        choice = match[0] if match[0] else match[1]
                        choices.append(choice)
                else:
                    # Try to split on newlines and clean up
                    choices_list = choices_raw.strip("[]").split("\n")
                    for item in choices_list:
                        item = item.strip()
                        if item and not item.isspace():
                            choices.append(item.strip())
            else:
                choices = choices_raw.split()
        
        # If we still have no choices, try other approaches
        if not choices:
            # Try to split by array indicators
            choices = str(choices_raw).strip("[]").split()
    except Exception as e:
        logging.error(f"Error parsing choices: {e}")
        print(f"Error parsing choices: {e}")
        choices = []
    
    return choices

def parse_response(response_text):
    """Parse the response to extract rewritten question, choices, and answer"""
    rewritten_question = None
    choices_text = None
    correct_answer = None
    
    # Remove thinking sections if present
    cleaned_response = response_text
    if "<think>" in cleaned_response and "</think>" in cleaned_response:
        think_start = cleaned_response.find("<think>")
        think_end = cleaned_response.find("</think>") + len("</think>")
        cleaned_response = cleaned_response[:think_start] + cleaned_response[think_end:]
        cleaned_response = cleaned_response.strip()
    
    # Try standard format
    if "Rewritten Question:" in cleaned_response:
        try:
            rq_start = cleaned_response.find("Rewritten Question:")
            ch_start = cleaned_response.find("Choices:", rq_start)
            ca_start = cleaned_response.find("Correct Answer:", ch_start)
            
            if rq_start != -1 and ch_start != -1:
                rewritten_question = cleaned_response[rq_start + len("Rewritten Question:"):ch_start].strip()
            
            if ch_start != -1 and ca_start != -1:
                choices_text = cleaned_response[ch_start + len("Choices:"):ca_start].strip()
            
            if ca_start != -1:
                correct_answer = cleaned_response[ca_start + len("Correct Answer:"):].strip()
        except Exception as e:
            print(f"Error parsing standard format: {e}")
    
    # Try line-by-line parsing if standard failed
    if not all([rewritten_question, choices_text, correct_answer]):
        try:
            lines = cleaned_response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Rewritten Question:"):
                    rewritten_question = line[len("Rewritten Question:"):].strip()
                elif line.startswith("Choices:"):
                    choices_text = line[len("Choices:"):].strip()
                elif line.startswith("Correct Answer:"):
                    correct_answer = line[len("Correct Answer:"):].strip()
        except Exception as e:
            print(f"Error with line-by-line parsing: {e}")
    
    # Last resort for rewritten question
    if not rewritten_question and "?" in cleaned_response:
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_response)
        for sentence in sentences:
            if sentence.strip().endswith("?"):
                rewritten_question = sentence.strip()
                break
    
    return rewritten_question, choices_text, correct_answer

def process_question(together_client, question_idx, question_info, output_file):
    """Process a single question and save to the output file"""
    try:
        question = question_info['question']
        subject = question_info['subject']
        
        # Parse choices and answer
        choices_raw = question_info['choices']
        choices = parse_choices(choices_raw)
        
        answer_idx = int(question_info['answer'])
        if 0 <= answer_idx < len(choices):
            answer = choices[answer_idx]
        else:
            answer = "Unknown"
        
        print(f"\nProcessing question {question_idx+1}/20: {subject}")
        print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
        print(f"Choices: {choices}")
        print(f"Correct answer: {answer}")
        
        # Create prompt
        prompt_template = """
You are an expert in question design. Rewrite the following MMLU question to maintain its topic, difficulty, and correct answer, but change the wording and structure. Ensure the question remains clear, concise, and has four answer choices with one correct answer.

Original Question: {question}
Choices: {choices}
Correct Answer: {answer}

Important: Provide only the rewritten question, choices, and correct answer with no commentary or thinking. Use this exact format:
Rewritten Question: [Your rewritten question]
Choices: ["Choice A", "Choice B", "Choice C", "Choice D"]
Correct Answer: [The correct answer text]
"""

        prompt = prompt_template.format(
            question=question,
            choices=choices,
            answer=answer
        )
        
        # Call API
        start_time = datetime.now()
        response = call_api(together_client, prompt)
        
        if response:
            response_text = response.choices[0].message.content.strip()
            
            # Save raw response
            os.makedirs("responses", exist_ok=True)
            with open(f"responses/response_raw_{question_idx}.txt", "w", encoding="utf-8") as f:
                f.write(response_text)
            
            # Parse response
            rewritten_question, choices_text, correct_answer = parse_response(response_text)
            
            # Use defaults if parsing failed
            if not rewritten_question:
                rewritten_question = f"Rewritten: {question}"
            
            if not choices_text:
                choices_text = str(choices)
            
            if not correct_answer:
                correct_answer = answer
            
            # Create result
            result = {
                "original_index": question_idx,
                "subject": subject,
                "original_question": question,
                "rewritten_question": rewritten_question,
                "choices": choices_text,
                "correct_answer": correct_answer
            }
            
            # Add to output CSV
            df_result = pd.DataFrame([result])
            if os.path.exists(output_file):
                df_result.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df_result.to_csv(output_file, index=False)
            
            print(f"Successfully processed question {question_idx+1}")
            duration = datetime.now() - start_time
            print(f"Processing time: {duration}")
            
            return True
            
        else:
            print(f"Failed to process question {question_idx+1}: No valid API response")
            return False
            
    except Exception as e:
        error_msg = f"Error processing question {question_idx+1}: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return False

def main():
    # Initialize Together client
    together_client = initialize_client()
    if not together_client:
        print("Failed to initialize client. Exiting.")
        return
    
    # Load the selected questions
    if not os.path.exists("selected_questions.csv"):
        print("Error: selected_questions.csv not found.")
        return
    
    # Define output file for new generation
    output_file = "mmlu_rewritten_samples_new.csv"
    if os.path.exists(output_file):
        print(f"Warning: {output_file} already exists. New results will be appended.")
    
    # Load questions
    questions_df = pd.read_csv("selected_questions.csv")
    total_questions = len(questions_df)
    print(f"Loaded {total_questions} questions to process")
    
    # Process each question
    start_time = datetime.now()
    print(f"Starting batch processing at {start_time}")
    
    success_count = 0
    failure_count = 0
    
    for idx in range(total_questions):
        print(f"\n{'='*50}")
        print(f"Processing question {idx+1}/{total_questions}")
        print(f"{'='*50}")
        
        question_info = questions_df.iloc[idx].to_dict()
        
        if process_question(together_client, idx, question_info, output_file):
            success_count += 1
        else:
            failure_count += 1
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"Processing completed at {end_time}")
    print(f"Total duration: {duration}")
    print(f"Questions processed: {success_count}/{total_questions}")
    print(f"Failed: {failure_count}")
    print(f"Results saved to {output_file}")
    print(f"{'='*50}")
    
    logging.info(f"Processing completed. Success: {success_count}, Failed: {failure_count}, Duration: {duration}")

if __name__ == "__main__":
    main() 