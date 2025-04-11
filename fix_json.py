# fix_json.py
import json
import os
import re

def fix_json_file(input_path, output_path):
    """Convert concatenated JSON objects into a valid JSON array."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Find individual JSON objects using regex
        objects = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', content)
        valid_objects = []
        for obj in objects:
            try:
                parsed = json.loads(obj)
                valid_objects.append(parsed)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid object in {input_path}: {e}")
        # Write as a valid JSON array
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(valid_objects, f, indent=2)
        print(f"Fixed JSON saved to {output_path}")
    except Exception as e:
        print(f"Error fixing {input_path}: {e}")

def main():
    input_dir = 'data/raw/'
    files = [
        'mutual_funds_data.json',
        'stock_data.json',
        'mf_holdings_data.json'
    ]
    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(input_dir, file)  # Overwrite original
        if os.path.exists(input_path):
            fix_json_file(input_path, output_path)
        else:
            print(f"File not found: {input_path}")

if __name__ == '__main__':
    main()