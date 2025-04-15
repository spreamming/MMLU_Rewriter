import os
import pandas as pd
import re
import time
import logging
import traceback
from together import Together
from dotenv import load_dotenv
from datetime import datetime
import shutil
import json
import sys
import random
 
# Set up logging
logging.basicConfig(
    filename='api_calls.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global timeout settings
API_TIMEOUT = 120  # API call timeout in seconds
PROCESSING_TIMEOUT = 300  # Overall processing timeout in seconds

# Constants for handling rate limits
MIN_DELAY_SECONDS = 0  # Changed from 120 to 0 - no delay
BACKOFF_FACTOR = 1     # Changed from 3 to 1 - no exponential backoff
MAX_RETRIES = 3        # Keep 3 retries
MAX_CUMULATIVE_WAIT = 0  # Changed from 1800 to 0 - no wait
FREE_TIER_QPM = 0.03  # Queries per minute allowed on free tier (1 query per ~33 minutes)
DEEPSEEK_R1_MODEL = "deepseek-ai/deepseek-coder-v2-instruct"

# API call function with retry mechanism
def call_deepseek_api(client, prompt, retry_count=0):
    # Remove wait time
    # wait_time = MIN_DELAY_SECONDS * (BACKOFF_FACTOR ** retry_count)
    
    print(f"\nCalling DeepSeek Distill API (attempt {retry_count + 1}/{MAX_RETRIES + 1})...")
    logging.info(f"Calling DeepSeek Distill API (attempt {retry_count + 1}/{MAX_RETRIES + 1})...")
    
    try:
        # Add a timeout to the API call to prevent hanging indefinitely
        response = client.chat.completions.create(
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
            retry_msg = f"Error occurred. Retrying immediately (attempt {next_retry + 1}/{MAX_RETRIES + 1})..."
            logging.warning(retry_msg)
            print(retry_msg)
            return call_deepseek_api(client, prompt, next_retry)
        else:
            raise e  # Re-raise if max retries exceeded or different error

def call_api_with_retry(together_client, model, prompt, max_tokens=1024, temperature=0.7, verbose=False):
    """
    Call the Together API with retry logic for rate limiting.
    
    Args:
        together_client: The Together client instance
        model: The model ID to use for generation
        prompt: The prompt to send to the model
        max_tokens: The maximum number of tokens to generate
        temperature: The temperature for generation
        verbose: Whether to print verbose logs
        
    Returns:
        The API response text, or None if all retries failed
    """
    retry_count = 0
    while retry_count <= MAX_RETRIES:
        try:
            # Remove wait time with exponential backoff
            if retry_count > 0 and verbose:
                print(f"Retrying API call (attempt {retry_count}/{MAX_RETRIES}) immediately...")
                logging.info(f"Retrying API call (attempt {retry_count}/{MAX_RETRIES}) immediately...")
            
            # Make the API call
            if verbose:
                print(f"Calling API with model: {model}")
            logging.info(f"Calling API with model: {model}, max_tokens: {max_tokens}, temperature: {temperature}")
            
            response = together_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # If we get here, the call was successful
            return response
            
        except Exception as e:
            error_str = str(e).lower()
            retry_count += 1
            
            # Log the error
            logging.error(f"API call failed: {error_str}")
            
            # Check if this is a rate limit error
            if "rate limit" in error_str or "429" in error_str:
                if retry_count <= MAX_RETRIES:
                    # Will retry
                    logging.warning(f"Rate limit exceeded. Will retry immediately ({retry_count}/{MAX_RETRIES})")
                    if verbose:
                        print(f"Rate limit exceeded. Will retry immediately ({retry_count}/{MAX_RETRIES})")
                else:
                    # Max retries reached
                    logging.error(f"Rate limit exceeded. Max retries ({MAX_RETRIES}) reached. Giving up.")
                    if verbose:
                        print(f"Rate limit exceeded. Max retries ({MAX_RETRIES}) reached. Giving up.")
                    return None
            else:
                # Not a rate limit error, don't retry
                logging.error(f"API call failed with non-rate-limit error: {error_str}")
                if verbose:
                    print(f"API call failed with error: {error_str}")
                return None
    
    # If we get here, all retries failed
    return None

def initialize_client():
    """
    Initialize the Together API client using the API key from environment variables.
    
    Returns:
        The initialized Together client, or None if initialization failed
    """
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

def load_and_select_questions():
    """
    Load questions from the selected_questions.csv file and filter to get questions
    that haven't been processed yet.
    
    Returns:
        List of question dictionaries to process
    """
    try:
        # Load the selected questions from CSV
        selected_file = "selected_questions.csv"
        if not os.path.exists(selected_file):
            raise FileNotFoundError(f"Selected questions file {selected_file} not found")
            
        selected_df = pd.read_csv(selected_file)
        print(f"Loaded {len(selected_df)} questions from {selected_file}")
        logging.info(f"Loaded {len(selected_df)} questions from {selected_file}")
        
        # Convert DataFrame rows to dictionaries
        all_questions = []
        for i, row in selected_df.iterrows():
            # Parse the choices from the string representation
            choices_str = row["choices"]
            # Handle choices that are in the format ['choice1' 'choice2' 'choice3' 'choice4']
            choices_str = choices_str.strip("[]").replace("'", "")
            choices = []
            
            # Parse based on newlines and spaces
            if "\n" in choices_str:
                # Split by newlines first
                choices_items = choices_str.split("\n")
                # Clean up each item
                for item in choices_items:
                    item = item.strip()
                    if item and not item.isspace():
                        if item.startswith("'") or item.startswith('"'):
                            item = item[1:]
                        if item.endswith("'") or item.endswith('"'):
                            item = item[:-1]
                        choices.append(item.strip())
            else:
                # Try to split by pattern recognition
                # Most choices are separated by quotes: 'choice1' 'choice2'
                import re
                matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", choices_str)
                if matches:
                    for match in matches:
                        # Each match is a tuple of groups, use the non-empty one
                        choice = match[0] if match[0] else match[1]
                        choices.append(choice)
                else:
                    # Fallback - split by spaces
                    choices = [c.strip() for c in choices_str.split() if c.strip()]
            
            # If we still have no choices, try a simple split
            if not choices:
                choices = [c.strip() for c in choices_str.split("' '")]
                
            # Get the answer based on the index
            answer_idx = int(row["answer"])
            answer = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
            
            question_dict = {
                "id": i,
                "original_index": i,
                "subject": row["subject"],
                "question": row["question"],
                "choices": choices,
                "answer": answer,
                "selected": True
            }
            all_questions.append(question_dict)
            
            # Debug
            print(f"\nProcessed question {i}: {row['subject']}")
            print(f"  Choices parsed: {choices}")
            print(f"  Answer index: {answer_idx}, Answer: {answer}")
        
        # Check if rewritten samples file exists
        if os.path.exists('mmlu_rewritten_samples.csv'):
            # Load already processed questions
            df = pd.read_csv('mmlu_rewritten_samples.csv')
            processed_indices = set(df['id'].tolist() if 'id' in df.columns else [])
            print(f"Found {len(processed_indices)} already processed questions.")
            logging.info(f"Found {len(processed_indices)} already processed questions.")
        else:
            processed_indices = set()
            print("No previously processed questions found.")
            logging.info("No previously processed questions found.")
        
        # Get the questions that haven't been processed yet
        questions_to_process = [q for q in all_questions if q['id'] not in processed_indices]
        print(f"Found {len(questions_to_process)} questions that need processing.")
        logging.info(f"Found {len(questions_to_process)} questions that need processing.")
        
        return questions_to_process
        
    except Exception as e:
        error_msg = f"Failed to load questions: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return []

def process_question(question_info, together_client):
    """
    Process a single question and save results to CSV
    """
    try:
        question_id = question_info.get("id", "unknown")
        subject = question_info.get("subject", "unknown")
        original_question = question_info.get("question", "")
        choices = question_info.get("choices", [])
        answer = question_info.get("answer", "")
        
        # Log question details
        logging.info(f"Processing question ID: {question_id}, Subject: {subject}")
        logging.info(f"Original question: {original_question}")
        logging.info(f"Choices: {choices}")
        logging.info(f"Correct answer: {answer}")
        
        print(f"\nProcessing question ID: {question_id}")
        print(f"Subject: {subject}")
        
        # Format prompt for rewriting
        prompt = (
            "Please rewrite the following multiple-choice question to make it more challenging while preserving its essence and correct answer.\n\n"
            f"Original Question: {original_question}\n\n"
            f"Choices:\n"
        )
        
        # Add choices to prompt
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        
        prompt += f"\nCorrect Answer: {answer}\n\n"
        prompt += (
            "Rewrite the question in the following JSON format:\n"
            "{\n"
            '  "rewritten_question": "Your rewritten question here",\n'
            '  "rewritten_choices": ["Choice A", "Choice B", "Choice C", "Choice D"],\n'
            '  "rewritten_answer": "The correct answer choice letter"\n'
            "}\n"
        )
        
        # Call the API with retries
        response = call_api_with_retry(
            together_client=together_client,
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            prompt=prompt,
            max_tokens=1024,
            temperature=0.3,
            verbose=True
        )
        
        # Extract the content from the API response
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content.strip()
            
            # Save raw response to a file
            os.makedirs("responses", exist_ok=True)
            with open(f"responses/response_raw_{question_id}.txt", "w", encoding="utf-8") as f:
                f.write(content)
            
            # Parse the JSON response
            try:
                # Find JSON content
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    json_content = json_match.group(1)
                    data = json.loads(json_content)
                    
                    # Extract data
                    rewritten_question = data.get('rewritten_question', '')
                    rewritten_choices = data.get('rewritten_choices', [])
                    rewritten_answer = data.get('rewritten_answer', '')
                    
                    # Validate rewritten data
                    if not rewritten_question or not rewritten_choices or not rewritten_answer:
                        raise ValueError("Incomplete data in API response")
                    
                    # Create result dictionary
                    result = {
                        "id": question_info.get("id", "unknown"),
                        "subject": subject,
                        "original_question": original_question,
                        "original_choices": choices,
                        "original_answer": answer,
                        "rewritten_question": rewritten_question,
                        "rewritten_choices": rewritten_choices,
                        "rewritten_answer": rewritten_answer
                    }
                    
                    # Save this individual result to avoid losing progress
                    save_individual_result(result, output_file="mmlu_rewritten_samples_new.csv")
                    
                    print(f"Successfully processed question ID: {question_id}")
                    logging.info(f"Successfully processed question ID: {question_id}")
                    
                    return result
                else:
                    error_msg = "Could not find JSON content in API response"
                    print(f"Error: {error_msg}")
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                    
            except json.JSONDecodeError as je:
                error_msg = f"Error parsing JSON from API response: {je}"
                print(f"Error: {error_msg}")
                logging.error(error_msg)
                logging.error(f"API response content: {content}")
                raise
                
            except Exception as e:
                error_msg = f"Error processing API response: {e}"
                print(f"Error: {error_msg}")
                logging.error(error_msg)
                raise
        else:
            error_msg = "Invalid or empty API response"
            print(f"Error: {error_msg}")
            logging.error(error_msg)
            raise ValueError(error_msg)
            
    except Exception as e:
        print(f"Error processing question {question_info.get('id', 'unknown')}: {e}")
        logging.error(f"Error processing question {question_info.get('id', 'unknown')}: {e}")
        # Re-raise the exception so it can be caught in the main function if needed
        raise

def save_individual_result(result, output_file="mmlu_rewritten_samples.csv"):
    """
    Save an individual result to the CSV file to prevent losing progress
    """
    try:
        if os.path.exists(output_file):
            # Check if this question is already in the file
            df = pd.read_csv(output_file)
            if any(df['id'] == result['id']):
                # Already saved, no need to append
                return
            
            # Append to existing file
            pd.DataFrame([result]).to_csv(output_file, mode='a', header=False, index=False)
        else:
            # Create new file
            pd.DataFrame([result]).to_csv(output_file, index=False)
        
        logging.info(f"Saved result for question with id {result['id']} to {output_file}")
        print(f"Saved result to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save individual result: {e}")
        print(f"Warning: Failed to save individual result: {e}")

def main():
    # Set up the Together client
    together_client = initialize_client()
    if not together_client:
        print("Failed to initialize Together client, exiting.")
        sys.exit(1)
    
    # Create responses directory
    os.makedirs("responses", exist_ok=True)
    
    # Output file
    output_file = "mmlu_rewritten_samples_new.csv"
    print(f"Will save results to {output_file}")
    
    # Load the MMLU dataset
    questions_to_process = load_and_select_questions()
    
    # Get the count of questions to process
    total_questions = len(questions_to_process)
    if total_questions == 0:
        print("No questions to process. Exiting.")
        return
    
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    print(f"Starting to process {total_questions} questions")
    logging.info(f"Starting to process {total_questions} questions")
    
    for i, question_info in enumerate(questions_to_process):
        try:
            question_id = question_info.get("id", "unknown")
            print(f"\n[{i+1}/{total_questions}] Processing question ID: {question_id}")
            
            # Process the question and save the result
            try:
                result = process_question(question_info, together_client)
                if result:
                    print(f"Successfully processed question {question_id}")
                    processed_count += 1
                else:
                    print(f"Skipped question {question_id} (result was None)")
                    skipped_count += 1
            except Exception as e:
                print(f"Failed to process question {question_id}: {str(e)}")
                logging.error(f"Failed to process question {question_id}: {str(e)}")
                failed_count += 1
            
            # No wait between questions
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving progress and exiting.")
            logging.info("Process interrupted by user. Saving progress and exiting.")
            break
    
    # Print summary
    print("\n===== Processing Summary =====")
    print(f"Total questions: {total_questions}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Results saved to {output_file}")
    logging.info(f"Processing complete. Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}")

if __name__ == "__main__":
    main() 