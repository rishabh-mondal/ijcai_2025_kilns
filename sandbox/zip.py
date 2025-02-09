import shutil
import os

# Define folder and zip output path
folder_path = "/home/rishabh.mondal/bkdb/statewise/rajasthan"
zip_output = "/home/rishabh.mondal/bkdb/statewise/rajasthan"

# Create a zip archive (without .zip extension in the output path)
shutil.make_archive(zip_output, 'zip', folder_path)

print(f"Zipped folder saved as: {zip_output}.zip")