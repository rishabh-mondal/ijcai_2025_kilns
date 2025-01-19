from SwinIR_wrapper.SwinIR_wrapper import SwinIR_SR

import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f'Using GPU: {torch.cuda.get_device_properties(0).name}')
else:
    print('Using CPU. Concider using GPU for faster inference.')



#@title Setup Super Resolution Model { run: "auto" }
pretrained_model = "real_sr x4" #@param ["real_sr x4", "classical_sr x2", "classical_sr x3", "classical_sr x4", "classical_sr x8", "lightweight x2", "lightweight x3", "lightweight x4"]

model_type, scale = pretrained_model.split(' ')
scale = int(scale[1])

# initialize super resolution model
sr = SwinIR_SR(model_type, scale)

print(f'Loaded {pretrained_model} successfully')

import os
import cv2
 # Replace with the actual module or model you're using

# Paths
input_dir = '../data/region_performance/test_bihar_same_class_count_10_120_1000/images'
output_dir = '../data/swinir_data/test_bihar_same_class_count_10_120_1000/images'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to process images
def process_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):  # Check for .tif files
            # Load the low-quality image
            img_lq_path = os.path.join(input_dir, filename)
            img_lq = cv2.imread(img_lq_path, cv2.IMREAD_COLOR)
            
            # Apply the super-resolution model
            img_hq = sr.upscale(img_lq)  # Adjust the method based on your SR model's API
            
            # Resize the image back to the original dimensions
            image_resized = cv2.resize(img_hq, (img_lq.shape[1], img_lq.shape[0]), interpolation=cv2.INTER_AREA)
            
            # Save the result as a .png file
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, image_resized)
            
            # Optional: Display or compare the images (for debugging or validation)
            # compare_sr_with_original(img_lq, image_resized)  # Implement or call your comparison function

# Execute the function
process_images(input_dir, output_dir)
