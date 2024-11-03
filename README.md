# Blood Cell Detection Web App

This repository contains a web application designed for detecting and classifying blood cells—specifically Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets—using a YOLOv10 model. The app allows users to upload images of blood samples and provides visualizations of the detected cells along with performance metrics.

## Project Structure

```
ML_intern_01_Artigence/
├── imagesets/
│   ├── train/         # Contains some training images used in training with YOLO annotations
│   └── val/           # Contains some validation images used in training with YOLO annotations
├── app.py             # Gradio interface for the trained YOLO model
├── bccd.yaml          # Configuration file for YOLOv10
├── convert.py         # Script to convert XML annotations to YOLO format
├── data_aug.py        # Data augmentation script using Albumentations
├── requirements.txt    # List of required Python packages
└── yolo.py            # Training script for the YOLO model
```

## Features

- **Image Upload**: Users can upload images of blood samples for detection.
- **Cell Detection**: Visualizes detected cells with bounding boxes.
- **Confidence Scores**: Displays confidence scores for each detected cell.
- **Performance Metrics**: Calculates and shows precision and recall metrics for the model.
- **User-Friendly Interface**: Built using Gradio for ease of use.

## Dataset
The dataset used for this project is the **BCCD Dataset**, consisting of over 410 original images. The images were then augmented and created over 2000+ images for efficient training. Each image has corresponding YOLO annotation files with the same name, located side by side in their respective folders.

### Annotation Format
- The annotations are in YOLO format, where each line in the annotation file corresponds to one object in the image, formatted as:
  ```
  class_id x_center y_center width height
  ```

### YAML Configuration
The `bccd.yaml` file contains the following configuration:
This configuration file specifies the paths to the training and validation image sets, the number of classes in the dataset, and the names of these classes (RBC, WBC, Platelets).


## Key Components

- **app.py**: This script creates a web interface using Gradio, allowing users to interact with the YOLO model. It displays predictions and outputs confidence scores, precision, and recall metrics using the Pandas library.
  
- **convert.py**: This script converts XML annotations to YOLO format. It is particularly useful for projects that previously used XML for annotations.
  
- **data_aug.py**: Implements data augmentation techniques to enhance the training dataset, primarily utilizing the Albumentations library.

- **requirements.txt**: Lists the necessary libraries used throughout the project, including:
  - `ultralytics`
  - `gradio`
  - `pandas`
  - `albumentations`
  - `Pillow`
  - `matplotlib`
  - `numpy`

- **yolo.py**: Contains the training logic for the YOLO model, including hyperparameter settings and model training routines, leveraging the Ultralytics library.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have your dataset in the specified directory structure as outlined in the **Project Structure** section.
2. Update the `bccd.yaml` file paths to point to your dataset location.
3. Run the training script:

```bash
python yolo.py
```

4. After training, start the Gradio interface:

```bash
python app.py
```
## Contributions

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



   

