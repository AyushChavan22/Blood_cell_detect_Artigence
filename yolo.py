import os
from ultralytics import YOLO

# This function generates a YAML configuration file that YOLOv10 uses to understand the structure of the BCCD dataset.
# It takes in the paths to the training and validation image directories and saves the YAML file at the specified path.
def create_bccd_yaml(yaml_path, train_dir, val_dir):
    # Define the YAML content, including the paths to training and validation data and the class labels.
    yaml_content = f"""
    # BCCD Dataset for YOLOv10
    train: {train_dir}  # Path to training images
    val: {val_dir}      # Path to validation images

    # Number of classes (we're detecting three types of cells)
    nc: 3

    # Class names corresponding to the classes in the dataset (RBC, WBC, and Platelets)
    names: ["RBC", "WBC", "Platelets"]
    """
    
    # Now we write this YAML content into a file, which YOLO will later use for training.
    with open(yaml_path, 'w') as file:
        file.write(yaml_content)
    
    # Inform the user that the configuration has been successfully saved.
    print(f"BCCD YAML configuration saved at: {yaml_path}")

# This function is responsible for loading the YOLO model and training it on the BCCD dataset.
# It uses the custom YAML configuration we created in the previous step.
def train_yolo_bccd(yaml_path):
    # First, let the user know that we’re about to load the YOLOv10 model.
    print("Loading the YOLOv10 model...")
    
    # Load the YOLOv10n model (the 'n' stands for 'nano' – a smaller version of YOLO, ideal for smaller datasets or limited computational resources).
    model = YOLO("yolov10n.yaml")  # We specify the YOLOv10n configuration here.
    
    # Now, inform the user that the training process is about to begin.
    print("Starting training on the BCCD dataset...")
    
    # Train the model using the custom YAML file and predefined training settings.
    try:
        model.train(
            data=yaml_path,       # The path to the custom BCCD YAML config we just created.
            epochs=60,            # The number of times the model will go through the entire dataset during training.
            imgsz=640,            # Image size – YOLO will resize all images to 640x640 pixels for consistent training.
            batch=4,              # The number of images the model processes at once (we set this low to match the capability of an i3 processor).
            name="yolov10_bccd",  # A custom name for this training run, helpful for organizing experiment results.
            pretrained=True,      # Start training using weights from a pre-trained YOLOv10 model, which speeds up training.
            lr0=0.01,             # The initial learning rate (how fast the model updates its knowledge).
            lrf=0.1,              # The final learning rate – how much the learning rate decays by the end of training.
            momentum=0.937,       # Momentum helps smooth out the training process (this controls how much of the previous update direction is retained).
            weight_decay=0.0005,  # Regularization to prevent the model from overfitting (making it perform well on training but poorly on new data).
            workers=2             # The number of CPU cores to use for loading data. We set this low due to hardware limitations (i3 processor).
        )
        # If training completes successfully, print a message indicating success.
        print("Training completed!")
    
    # Catch any potential errors during training and print them.
    except Exception as e:
        print(f"Error during training: {e}")

# This block runs when the script is executed directly.
if __name__ == "__main__":
    # Set the paths to the training and validation image directories (adjust these paths as necessary).
    train_dir = r"C:\Users\AYUSH\OneDrive - Vidyalankar Institute of Technology\Desktop\ML_INTERN_01\BCCD_Dataset\BCCD\ImageSets\Main\train"  # Adjust this path to your train images folder
    val_dir = r"C:\Users\AYUSH\OneDrive - Vidyalankar Institute of Technology\Desktop\ML_INTERN_01\BCCD_Dataset\BCCD\ImageSets\Main\val"      # Adjust this path to your validation images folder
    
    # Set the path where the custom YAML file will be saved.
    yaml_path = "bccd_custom.yaml"
    
    # Step 1: Create the custom YAML file for the BCCD dataset, which YOLOv10 will use during training.
    create_bccd_yaml(yaml_path, train_dir, val_dir)
    
    # Step 2: Train YOLOv10 on the BCCD dataset using the custom YAML config and the specified training parameters.
    train_yolo_bccd(yaml_path)
