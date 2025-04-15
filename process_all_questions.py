import os
import pandas as pd
import time
import sys
import signal
import logging
import traceback
from datetime import datetime
from subprocess import run, PIPE, TimeoutExpired

# Set up logging
logging.basicConfig(
    filename='question_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define constants for rate limiting
MIN_BETWEEN_QUESTIONS_SECONDS = 360  # Wait 6 minutes between question processing attempts
SCRIPT_TIMEOUT = 600  # 10 minutes timeout for each question processing

def get_checkpoint():
    """Read the checkpoint file to determine where to start processing."""
    if os.path.exists("processing_checkpoint.txt"):
        with open("processing_checkpoint.txt", "r") as f:
            try:
                return int(f.read().strip())
            except:
                return 0
    return 0

def save_checkpoint(idx):
    """Save the current processing index to the checkpoint file."""
    with open("processing_checkpoint.txt", "w") as f:
        f.write(str(idx))
    logging.info(f"Checkpoint saved: {idx}")

def main():
    start_time = datetime.now()
    logging.info(f"Script started at {start_time}")
    
    # Check if selected_questions.csv exists
    if not os.path.exists("selected_questions.csv"):
        msg = "Error: selected_questions.csv not found."
        logging.error(msg)
        print(msg)
        return
    
    # Get number of questions to process
    selected_df = pd.read_csv("selected_questions.csv")
    total_questions = len(selected_df)
    logging.info(f"Found {total_questions} questions to process")
    print(f"Found {total_questions} questions to process")
    
    # Check for existing results
    processed_indices = []
    if os.path.exists("mmlu_rewritten_samples.csv"):
        results_df = pd.read_csv("mmlu_rewritten_samples.csv")
        if 'original_index' in results_df.columns:
            processed_indices = [int(idx) for idx in results_df['original_index'].tolist()]
            msg = f"Found {len(processed_indices)} already processed questions: {processed_indices}"
            logging.info(msg)
            print(msg)
    
    # Get the starting index from checkpoint
    start_idx = get_checkpoint()
    logging.info(f"Starting from question index {start_idx}")
    print(f"Starting from question index {start_idx}")
    
    # Process questions sequentially
    for idx in range(start_idx, total_questions):
        if idx in processed_indices:
            msg = f"Question {idx} already processed, skipping..."
            logging.info(msg)
            print(msg)
            save_checkpoint(idx + 1)  # Update checkpoint to next question
            continue
        
        # Run the generate_next_sample.py script with the specific index
        separator = "=" * 50
        msg = f"\n{separator}\nProcessing question {idx}...\n{separator}"
        logging.info(msg)
        print(msg)
        
        try:
            # Modify the script to run the next sample
            with open("generate_next_sample.py", "r") as f:
                script_content = f.read()
            
            # Find the line where the next_idx is set and replace it
            new_script = []
            for line in script_content.split("\n"):
                if line.strip().startswith("next_idx = "):
                    new_script.append(f"next_idx = {idx}  # Set to process question {idx}")
                else:
                    new_script.append(line)
            
            # Write back the modified script
            with open("generate_next_sample.py", "w") as f:
                f.write("\n".join(new_script))
            
            # Run the script with a timeout
            try:
                msg = f"Running script with {SCRIPT_TIMEOUT} second timeout..."
                logging.info(msg)
                print(msg)
                
                result = run(["python", "generate_next_sample.py"], stdout=PIPE, stderr=PIPE, text=True, timeout=SCRIPT_TIMEOUT)
                
                print(result.stdout)
                logging.info(f"Script output: {result.stdout}")
                
                if result.stderr:
                    msg = f"ERRORS: {result.stderr}"
                    logging.error(msg)
                    print(msg)
                    
            except TimeoutExpired:
                msg = f"ERROR: Script execution timed out after {SCRIPT_TIMEOUT} seconds!"
                logging.error(msg)
                print(msg)
                print("Moving to next question...")
            
            # Check if the script was successful
            if os.path.exists("mmlu_rewritten_samples.csv"):
                results_df = pd.read_csv("mmlu_rewritten_samples.csv")
                if 'original_index' in results_df.columns and idx in results_df['original_index'].tolist():
                    msg = f"Successfully processed question {idx}"
                    logging.info(msg)
                    print(msg)
                    # Update processed indices
                    processed_indices.append(idx)
                else:
                    msg = f"Failed to process question {idx}, will retry next time"
                    logging.warning(msg)
                    print(msg)
            
            # Save checkpoint after processing this question
            save_checkpoint(idx + 1)
            
            # Wait before processing next question
            if idx < total_questions - 1:
                msg = f"\nWaiting {MIN_BETWEEN_QUESTIONS_SECONDS} seconds before processing next question..."
                logging.info(msg)
                print(msg)
                time.sleep(MIN_BETWEEN_QUESTIONS_SECONDS)
                
        except Exception as e:
            error_msg = f"Unexpected error processing question {idx}: {str(e)}"
            stack_trace = traceback.format_exc()
            logging.error(f"{error_msg}\n{stack_trace}")
            print(f"ERROR: {error_msg}")
            print("Moving to next question...")
            # Don't update checkpoint so we retry this question next time
    
    # Script completed
    end_time = datetime.now()
    duration = end_time - start_time
    msg = f"Script completed at {end_time}. Total duration: {duration}"
    logging.info(msg)
    print(msg)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user")
        print("\nScript interrupted by user. Progress has been saved.")
    except Exception as e:
        error_msg = f"Unhandled error: {str(e)}"
        stack_trace = traceback.format_exc()
        logging.error(f"{error_msg}\n{stack_trace}")
        print(f"ERROR: {error_msg}")
        print("The script will exit. Check the log file for details.") 