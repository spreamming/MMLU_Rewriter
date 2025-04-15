import os
import pandas as pd
import re
import time
import logging
import traceback
from together import Together
from dotenv import load_dotenv
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='api_calls.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global timeout settings
API_TIMEOUT = 120  # API call timeout in seconds
PROCESSING_TIMEOUT = 300  # Overall processing timeout in seconds

# Define rate limits based on tier
MIN_DELAY_SECONDS = 30  # Reduced wait time (30 seconds)
MAX_RETRIES = 3  # Maximum number of retry attempts
BACKOFF_FACTOR = 2  # Exponential backoff factor

# API call function with retry mechanism
def call_deepseek_api(client, prompt, retry_count=0):
    wait_time = MIN_DELAY_SECONDS * (BACKOFF_FACTOR ** retry_count)
    print(f"\nWaiting {wait_time} seconds before API call to respect rate limits...")
    logging.info(f"Waiting {wait_time} seconds before API call to respect rate limits...")
    time.sleep(wait_time)
    
    print(f"Calling DeepSeek Distill API (attempt {retry_count + 1}/{MAX_RETRIES + 1})...")
    logging.info(f"Calling DeepSeek Distill API (attempt {retry_count + 1}/{MAX_RETRIES + 1})...")
    
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  # I used a smaller model 
            messages=[{
                "role": "system", 
                "content": "You are an expert in question design. Follow instructions exactly and provide only what is requested."
            }, {
                "role": "user", 
                "content": prompt
            }],
            max_tokens=500,
            temperature=0.5,  # Lower temperature for more consistent output
            timeout=API_TIMEOUT  # Add a timeout
        )
        return response
    except Exception as e:
        error_message = str(e)
        error_msg = f"Error calling API: {error_message}"
        logging.error(error_msg)
        print(error_msg)
        
        # Check if it's a rate limit error
        if "rate_limit" in error_message and retry_count < MAX_RETRIES:
            next_retry = retry_count + 1
            retry_msg = f"Rate limit exceeded. Retrying in {wait_time * BACKOFF_FACTOR} seconds (attempt {next_retry + 1}/{MAX_RETRIES + 1})..."
            logging.warning(retry_msg)
            print(retry_msg)
            return call_deepseek_api(client, prompt, next_retry)
        else:
            raise e  # Re-raise if max retries exceeded or different error

def process_question(question_idx):
    try:
        # Load environment variables
        load_dotenv()
        TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        if not TOGETHER_API_KEY:
            msg = "API Key not found. Please check your .env file."
            logging.error(msg)
            print(msg)
            return False

        print("API Key loaded successfully")
        logging.info("API Key loaded successfully")

        # Initialize Together AI client
        client = Together(api_key=TOGETHER_API_KEY)
        print("Together client initialized")
        logging.info("Together client initialized")

        # Find next question to process
        selected_df = pd.read_csv("selected_questions.csv")
        total_questions = len(selected_df)
        print(f"Found {total_questions} selected questions")
        logging.info(f"Found {total_questions} selected questions")

        # Check existing results
        processed_indices = []
        if os.path.exists("mmlu_rewritten_samples.csv"):
            results_df = pd.read_csv("mmlu_rewritten_samples.csv")
            if 'original_index' in results_df.columns:
                processed_indices = [int(idx) for idx in results_df['original_index'].tolist()]
            print(f"Found {len(processed_indices)} already processed questions")
            logging.info(f"Found {len(processed_indices)} already processed questions: {processed_indices}")

        # Check if this question has already been processed
        if question_idx in processed_indices:
            print(f"Question {question_idx} already processed, skipping...")
            logging.info(f"Question {question_idx} already processed, skipping...")
            return True

        # Get the question to process
        row = selected_df.iloc[question_idx]
        question = row['question']
        choices_raw = row['choices']
        answer_idx = row['answer']

        # Print question details
        print(f"\nSubject: {row['subject']}")
        print(f"Question: {question[:100]}..." if len(question) > 100 else question)
        logging.info(f"Processing subject: {row['subject']} (question {question_idx})")

        # Parse choices properly
        choices = []
        try:
            # Try to parse the choices from raw format
            if isinstance(choices_raw, str):
                if "'" in choices_raw:
                    matches = re.findall(r"'([^']*)'", choices_raw)
                    if matches:
                        choices = matches
                else:
                    choices = choices_raw.split()
            else:
                choices = choices_raw
            
            # If we still have no choices, try another approach
            if not choices:
                choices = str(choices_raw).split()
        except Exception as e:
            error_msg = f"Error parsing choices: {e}"
            logging.error(error_msg)
            print(error_msg)
            print("Using choices as-is")
            choices = choices_raw

        print(f"Parsed choices: {choices}")
        logging.info(f"Parsed choices: {choices}")

        # Get correct answer
        try:
            answer = choices[answer_idx]
            print(f"Answer: {answer}")
            logging.info(f"Answer: {answer}")
        except Exception as e:
            error_msg = f"Error getting answer: {e}"
            logging.error(error_msg)
            print(error_msg)
            print("Setting answer to first choice")
            answer = choices[0] if choices else "Unknown"
            logging.info(f"Defaulted answer to: {answer}")

        # Format prompt
        prompt_template = """
You are an expert in question design. Rewrite the following MMLU question to maintain its topic, difficulty, and correct answer, but change the wording and structure. Ensure the question remains clear, concise, and has four answer choices with one correct answer.

Original Question: {question}
Choices: {choices}
Correct Answer: {answer}

Important: Provide only the rewritten question, choices, and correct answer with no commentary or thinking. Use this exact format:
Rewritten Question: In 1962, President Kennedy declared, "We choose to go to the moon in this decade and do the other things, not because they are easy, but because they are hard." Which national initiative was most directly influenced by this famous address?
Choices: ["The Vietnam War", "The Mutually Assured Destruction (MAD) nuclear strategy", "The Apollo space program", "The Great Society social programs"]
Correct Answer: The Apollo space program
"""

        prompt = prompt_template.format(
            question=question,
            choices=choices,
            answer=answer
        )

        # Call DeepSeek API with retry mechanism
        print("Attempting to call DeepSeek Distill API with retry mechanism...")
        logging.info("Attempting to call DeepSeek Distill API with retry mechanism...")
        
        start_time = datetime.now()
        logging.info(f"Processing started at {start_time}")
        
        try:
            response = call_deepseek_api(client, prompt)
            
            # Get response content
            response_text = response.choices[0].message.content.strip()
            print("\nAPI RESPONSE RECEIVED!")
            logging.info("API RESPONSE RECEIVED!")
            
            # Save raw response to a file
            with open(f"response_raw_{question_idx}.txt", "w") as f:
                f.write(response_text)
            print(f"Raw response saved to response_raw_{question_idx}.txt")
            logging.info(f"Raw response saved to response_raw_{question_idx}.txt")
            
            # Print full response for debugging
            print("\nFull API Response:")
            print(response_text)
            
            # Enhanced parsing logic
            rewritten_question = None
            choices_text = None
            correct_answer = None
            
            # Remove any thinking sections
            cleaned_response = response_text
            if "<think>" in cleaned_response and "</think>" in cleaned_response:
                think_start = cleaned_response.find("<think>")
                think_end = cleaned_response.find("</think>") + len("</think>")
                cleaned_response = cleaned_response[:think_start] + cleaned_response[think_end:]
                cleaned_response = cleaned_response.strip()
                logging.info("Removed thinking section from response")
            
            # Try standard format first
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
                    
                    logging.info("Parsed response using standard format")
                except Exception as e:
                    error_msg = f"Error with standard parsing: {e}"
                    logging.error(error_msg)
                    print(error_msg)
            
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
                    
                    logging.info("Parsed response using line-by-line approach")
                except Exception as e:
                    error_msg = f"Error with line-by-line parsing: {e}"
                    logging.error(error_msg)
                    print(error_msg)
            
            # Last resort: look for specific formatting
            if not rewritten_question and "?" in cleaned_response:
                # Find the first sentence ending with a question mark
                sentences = re.split(r'(?<=[.!?])\s+', cleaned_response)
                for sentence in sentences:
                    if sentence.strip().endswith("?"):
                        rewritten_question = sentence.strip()
                        logging.info("Parsed question using question mark detection")
                        break
            
            # Set defaults if parsing failed
            if not rewritten_question:
                print("Could not extract rewritten question. Using default.")
                logging.warning("Could not extract rewritten question. Using default.")
                rewritten_question = f"Rewritten: {question}"
            
            if not choices_text:
                print("Could not extract choices. Using original choices.")
                logging.warning("Could not extract choices. Using original choices.")
                choices_text = str(choices)
            
            if not correct_answer:
                print("Could not extract correct answer. Using original answer.")
                logging.warning("Could not extract correct answer. Using original answer.")
                correct_answer = answer
            
            # Save to CSV
            result = {
                "original_index": question_idx,
                "subject": row['subject'],
                "original_question": question,
                "rewritten_question": rewritten_question,
                "choices": choices_text,
                "correct_answer": correct_answer
            }
            
            # Load existing results
            if os.path.exists("mmlu_rewritten_samples.csv"):
                results_df = pd.read_csv("mmlu_rewritten_samples.csv")
            else:
                results_df = pd.DataFrame()
            
            # Add new result
            if len(results_df) == 0:
                results_df = pd.DataFrame([result])
            else:
                results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
            
            # Save to CSV
            results_df.to_csv("mmlu_rewritten_samples.csv", index=False)
            print(f"Results saved to mmlu_rewritten_samples.csv")
            logging.info(f"Results saved to mmlu_rewritten_samples.csv")
            
            # Display the rewritten question
            print("\nRewritten Question:")
            print(rewritten_question)
            print(f"Choices: {choices_text}")
            print(f"Correct Answer: {correct_answer}")
            print("\nProcessing completed successfully!")
            
            end_time = datetime.now()
            duration = end_time - start_time
            logging.info(f"Processing completed successfully at {end_time}. Duration: {duration}")
            return True
            
        except Exception as e:
            end_time = datetime.now()
            duration = end_time - start_time
            error_msg = f"Error during processing: {str(e)}"
            stack_trace = traceback.format_exc()
            logging.error(f"{error_msg}\n{stack_trace}")
            logging.error(f"Processing failed after {duration}")
            print(error_msg)
            print("\nPlease try again.")
            return False
    
    except Exception as e:
        error_msg = f"Unhandled error in script: {str(e)}"
        stack_trace = traceback.format_exc()
        if 'logging' in globals():
            logging.error(f"{error_msg}\n{stack_trace}")
        print(error_msg)
        print("\nScript execution failed.")
        return False

def process_all_remaining_questions():
    # Check for checkpoint
    checkpoint = 0
    if os.path.exists("processing_checkpoint.txt"):
        with open("processing_checkpoint.txt", "r") as f:
            try:
                checkpoint = int(f.read().strip())
            except:
                checkpoint = 0
    
    print(f"Starting from checkpoint: {checkpoint}")
    
    # Get total questions
    selected_df = pd.read_csv("selected_questions.csv")
    total_questions = len(selected_df)
    
    # Get already processed questions
    processed_indices = []
    if os.path.exists("mmlu_rewritten_samples.csv"):
        results_df = pd.read_csv("mmlu_rewritten_samples.csv")
        if 'original_index' in results_df.columns:
            processed_indices = [int(idx) for idx in results_df['original_index'].tolist()]
    
    # Process remaining questions
    for idx in range(checkpoint, total_questions):
        if idx in processed_indices:
            print(f"Question {idx} already processed, skipping...")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing question {idx}...")
        print(f"{'='*50}")
        
        success = process_question(idx)
        
        # Update checkpoint
        with open("processing_checkpoint.txt", "w") as f:
            f.write(str(idx + 1))
        
        if idx < total_questions - 1:
            wait_time = 60  # Reduced wait time to 1 minute
            print(f"\nWaiting {wait_time} seconds before processing next question...")
            time.sleep(wait_time)

if __name__ == "__main__":
    # Process all remaining questions
    process_all_remaining_questions() 