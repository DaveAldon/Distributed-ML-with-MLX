import json
import random
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Split a JSONL file into train and valid sets.')
parser.add_argument('input_file', type=str, help='Path to the input JSONL file')
parser.add_argument('train_file', type=str, help='Path to the output train JSONL file')
parser.add_argument('valid_file', type=str, help='Path to the output valid JSONL file')
args = parser.parse_args()

# Read the input file
with open(args.input_file, 'r') as f:
    lines = f.readlines()

# Shuffle the lines
random.shuffle(lines)

# Calculate the split index
split_index = int(0.8 * len(lines))

# Split the lines into train and valid sets
train_lines = lines[:split_index]
valid_lines = lines[split_index:]

def format_line(line):
    data = json.loads(line)
    formatted_data = {
        "messages": [
            {
                "role": "user",
                "content": f"You are a helpful assistant.\n\n{data['instruction']}\n\nContext: {data['context']}"
            },
            {
                "role": "assistant",
                "content": data['response']
            }
        ]
    }
    return json.dumps(formatted_data)

# Write the train lines to train.jsonl
with open(args.train_file, 'w') as f:
    for line in train_lines:
        f.write(format_line(line) + '\n')

# Write the valid lines to valid.jsonl
with open(args.valid_file, 'w') as f:
    for line in valid_lines:
        f.write(format_line(line) + '\n')

print(f"Split {len(lines)} lines into {len(train_lines)} train and {len(valid_lines)} valid lines.")