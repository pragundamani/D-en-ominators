import os
import shutil
import re

base_path = os.getcwd()

# Regex that extracts letters + digits at the START
pattern = re.compile(r"^([A-Za-z]+\d+)\b")

for file in os.listdir(base_path):

    # Only process PNG files
    if not file.lower().endswith(".png"):
        continue

    file_path = os.path.join(base_path, file)

    # Skip directories
    if os.path.isdir(file_path):
        continue

    # Remove .png extension
    name_no_ext = file[:-4]  # e.g. "W24 (8)"

    # Extract base name using regex
    match = pattern.match(name_no_ext)
    if not match:
        print(f"Skipping (pattern not matched): {file}")
        continue

    base_name = match.group(1)   # "W24"

    # Create folder if needed
    target_folder = os.path.join(base_path, base_name)
    os.makedirs(target_folder, exist_ok=True)

    # Move file
    print(f"Moving {file} → {target_folder}")
    shutil.move(file_path, os.path.join(target_folder, file))

print("\nDone.")
