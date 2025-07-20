import os
from ultralytics import YOLO

# Load model
model = YOLO('Model/Yolo/chart_yolo2_best.pt')

# Input image path
input_image_path = 'Data/test/png/34.png'

# Extract image filename
image_filename = os.path.basename(input_image_path)  

# Define output path using the same name
output_path = os.path.join('Model/Yolo/Output', image_filename)

# Run inference
results = model(input_image_path)

# Show the result
results[0].show()

# Save the result with the same filename in the output directory
results[0].save(filename=output_path)
