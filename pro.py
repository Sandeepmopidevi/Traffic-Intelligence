import os

# Define the root directory
root_dir = r'S:\Temp\Traffic-Intelligence'

# Folders to exclude
exclude_folders = {'.git', 'node_modules', '.vs_code'}

def extract_file_paths(directory):
    file_paths = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(directory):
        # Exclude specified folders
        dirnames[:] = [d for d in dirnames if d not in exclude_folders]
        
        # Collect file paths
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))

    return file_paths

# Call the function to extract file paths
file_paths = extract_file_paths(root_dir)

# Print the file paths
for path in file_paths:
    print(path)
