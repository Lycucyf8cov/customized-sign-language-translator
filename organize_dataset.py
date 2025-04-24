import os
import shutil

# Set your dataset path here
dataset_path = os.path.join(os.getcwd(), 'dataset')

# Create subfolders for each class (A-Z)
for file in os.listdir(dataset_path):
    if os.path.isfile(os.path.join(dataset_path, file)):
        label = file[0].upper()  # Assuming file name starts with label (e.g., A_img1.jpg)
        label_folder = os.path.join(dataset_path, label)

        # Create the folder if it doesn't exist
        os.makedirs(label_folder, exist_ok=True)

        # Move the file into its label folder
        src = os.path.join(dataset_path, file)
        dst = os.path.join(label_folder, file)
        shutil.move(src, dst)

print("✅ Dataset organized into folders successfully!")
