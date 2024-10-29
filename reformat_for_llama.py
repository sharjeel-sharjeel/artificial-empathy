import json
import os

def create_jsonl_from_conversations(input_file_path, output_file):
    with open(input_file_path, 'r') as infile:
        # Read the lines two by two (output then input)
        while True:
            output_line = infile.readline()
            input_line = infile.readline()

            # Break if we reach the end of file
            if not output_line or not input_line:
                break

            # Extracting the conversation text and stripping newline characters
            output_text = output_line.replace('Output: ', '').strip()
            input_text = input_line.replace('Input: ', '').strip()

            # Create a dictionary with the input and output
            dialogue_turn = {
                "output": output_text,
                "input": input_text
            }

            # Convert the dictionary to a JSON string and write to file
            json_line = json.dumps(dialogue_turn)
            output_file.write(json_line + '\n')

def process_all_txt_files(directory_path, output_file_path):
    # Open the output file once and pass the file object to the function
    with open(output_file_path, 'w') as outfile:
        # Iterate over all files in the given directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                # Construct full file path
                file_path = os.path.join(directory_path, filename)
                # Process each file and append to the single jsonl file
                create_jsonl_from_conversations(file_path, outfile)

# Example usage:
directory_path = 'Train_Text/xyz'  # Replace with your directory containing the .txt files
output_file_path = 'Train_Text/lets_see.jsonl'  # Replace with the desired output file path

process_all_txt_files(directory_path, output_file_path)


import pandas as pd

def convert_parquet(input_file, output_file, output_format='csv'):
    """
    Convert a Parquet file to CSV or JSON.

    Parameters:
    input_file (str): Path to the input Parquet file.
    output_file (str): Path to the output file.
    output_format (str): The format of the output file ('csv' or 'json').
    """
    # Read the Parquet file
    df = pd.read_parquet(input_file)

    # Convert and save to the desired format
    if output_format == 'csv':
        df.to_csv(output_file, index=False)
    elif output_format == 'json':
        df.to_json(output_file, orient='records', lines=True)

# Example usage:
# convert_parquet('path/to/your/input_file.parquet', 'path/to/your/output_file.csv', 'csv')
# or
convert_parquet('train-00000-of-00001-6ef3991c06080e14.parquet', 'convertedfromparquet.json', 'json')
