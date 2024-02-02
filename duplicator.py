def duplicate_text_in_file(input_file_path, output_file_path, n):
    """
    Reads text from input_file_path, duplicates it n times, and writes the result to output_file_path.
    """
    try:
        with open(input_file_path, 'r') as file:
            content = file.read()
        
        duplicated_content = (content + '\n') * n
        
        with open(output_file_path, 'w') as file:
            file.write(duplicated_content)
        
        print(f"Content duplicated successfully {n} times and written to {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

input_file_path = 'base_input.txt'
output_file_path = 'duplicated_input.txt'
n = 5

duplicate_text_in_file(input_file_path, output_file_path, n)
