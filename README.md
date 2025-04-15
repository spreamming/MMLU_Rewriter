# MMLU Question Rewriter

This project focuses on rewriting questions from the MMLU (Massive Multitask Language Understanding) dataset using the DeepSeek-R1 language model. The goal is to create more challenging variants of the original questions while preserving their core concepts and correct answers.

## Project Overview

The MMLU dataset contains questions across various domains of knowledge. This project:

1. Selects a subset of 20 questions from the MMLU dataset
2. Uses the DeepSeek-R1 model to rewrite these questions
3. Preserves each question's subject area, difficulty level, and correct answer
4. Outputs rewritten questions in a structured format

## Repository Structure

- `generate_next_sample.py` - Main script for processing questions
- `generate_all_questions_fast.py` - Faster version without rate limit handling
- `process_all_questions.py` - Batch processor for all selected questions
- `process_single_fast.py` - Processes a single question quickly
- `question_processor.py` - Core processing logic
- `MMLUwithDeepseek.py` - Script for selecting MMLU questions
- `selected_questions.csv` - Contains the 20 selected MMLU questions
- `mmlu_rewritten_samples.csv` - Contains the rewritten questions
- `simulate_results.py` - For simulation purposes
- `requirements.txt` - Project dependencies

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Together API key:
   ```
   TOGETHER_API_KEY=your_api_key_here
   ```

## Usage

### Step 1: Select questions from MMLU dataset
```
python MMLUwithDeepseek.py
```
This will select 20 questions from the MMLU dataset and save them to `selected_questions.csv`.

### Step 2: Process the selected questions
```
python generate_next_sample.py
```
This will process the selected questions using the DeepSeek-R1 model and save the results to `mmlu_rewritten_samples.csv`.

### Alternative: Process all questions quickly
```
python generate_all_questions_fast.py
```
This version processes all questions without waiting between API calls (useful for smaller models).

## Sample Output

The rewritten questions preserve the core concept and correct answer of the original question but change the wording and structure to make it more challenging.

## Dependencies

- Python 3.8+
- together-python (for Together AI API)
- pandas
- python-dotenv

## License

This project is open source and available under the MIT License. 