from PIL import Image, ImageDraw, ImageFont  # Libraries for image manipulation and drawing
from ultralytics import YOLO  # YOLO library for object detection
import pandas as pd  # Pandas for working with dataframes
import gradio as gr  # Gradio for building the web interface

# This function loads the YOLOv10n model, which is the neural network trained for detecting blood cells.
def load_model(model_path):
    # Load the model using the provided path to the trained weights.
    model = YOLO(model_path)
    return model

# Path to the pre-trained model's weights. Adjust this to the location of your trained model.
model_path = r'C:\Users\AYUSH\OneDrive - Vidyalankar Institute of Technology\Desktop\ML_INTERN_01\runs\detect\yolov10_bccd\weights\best.pt' 
model = load_model(model_path)  # Load the YOLOv10n model.

# Define a color mapping for each class (RBC, WBC, and Platelets).
# This is used to draw bounding boxes in different colors for easy distinction between cell types.
COLORS = {
    "RBC": "red",       # RBC (Red Blood Cell) will have red boxes
    "WBC": "blue",      # WBC (White Blood Cell) will have blue boxes
    "Platelets": "brown"  # Platelets will have brown boxes
}

# This function performs predictions on the input image using the YOLO model and draws bounding boxes around detected cells.
def predict(model, image):
    # Use the YOLO model to make predictions on the input image.
    results = model(image)  
    
    # Extract bounding box coordinates (xmin, ymin, xmax, ymax) from the YOLO result.
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # Extract the class labels (RBC, WBC, Platelets) predicted for each detected object.
    classes = results[0].boxes.cls.cpu().numpy()
    
    # Extract the confidence scores for each prediction (how certain the model is about the detection).
    confidences = results[0].boxes.conf.cpu().numpy()
    
    # Get the class labels defined by the YOLO model (the names of the detected objects).
    labels = model.names  
    
    # Create a drawing object to draw on the image.
    draw = ImageDraw.Draw(image)
    
    # Load a font to write the class labels and confidence scores on the image.
    # You can adjust the font size to make it larger or smaller.
    font = ImageFont.truetype("arial.ttf", size=20)
    
    predictions = []  # Initialize an empty list to store predictions (label and confidence).

    # Iterate through the bounding boxes, classes, and confidence scores for each detection.
    for box, cls, conf in zip(boxes, classes, confidences):
        xmin, ymin, xmax, ymax = box  # Coordinates of the bounding box.
        label = labels[int(cls)]  # Get the class label using the class index.
        
        # Choose the color corresponding to the detected class (RBC, WBC, or Platelets).
        color = COLORS.get(label, "white")  # Default to white if the class isn't found.
        
        # Draw the bounding box on the image.
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        
        # Create the text to display the class label and confidence score.
        label_text = f"{label}: {conf:.2f}"
        
        # Get the text bounding box size to create a background for the label.
        text_bbox = draw.textbbox((xmin, ymin), label_text, font=font)
        
        # Draw a rectangle behind the text to make it more readable.
        draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill=color)
        
        # Draw the class label and confidence score on the image.
        draw.text((xmin, ymin), label_text, fill='white', font=font)
        
        # Store the prediction (label and confidence score) in the list for further analysis.
        predictions.append((label, conf))

    # Return the image with drawn bounding boxes and the list of predictions.
    return image, predictions

# This function calculates precision and recall for each class based on the predictions made by the model.
def calculate_metrics(predictions):
    # If no predictions were made, return an empty table.
    if not predictions:
        return {"Class": [], "Precision": [], "Recall": []}
    
    # Convert the predictions (class and confidence) into a pandas DataFrame for easier calculation.
    df = pd.DataFrame(predictions, columns=["Class", "Confidence"])
    
    metrics = {}  # Initialize an empty dictionary to store precision and recall for each class.
    total_tp, total_fp, total_fn = 0, 0, 0  # Variables for overall metrics (true positives, false positives, false negatives).

    # For each cell type (RBC, WBC, Platelets), calculate precision and recall.
    for cls in COLORS.keys():
        # Count true positives (correct detections with confidence >= 0.5).
        tp = sum((df["Class"] == cls) & (df["Confidence"] >= 0.5))
        # Count false positives (incorrect detections with confidence >= 0.5).
        fp = sum((df["Class"] != cls) & (df["Confidence"] >= 0.5))
        # Count false negatives (missed detections or low-confidence detections).
        fn = sum((df["Class"] == cls) & (df["Confidence"] < 0.5))
        
        # Update the overall metrics counters.
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Calculate precision for the current class (handling division by zero).
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        # Calculate recall for the current class.
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Store the calculated precision and recall in the metrics dictionary.
        metrics[cls] = {"Precision": precision, "Recall": recall}
    
    # Prepare a table to store precision and recall for each class.
    result_table = {"Class": [], "Precision": [], "Recall": []}
    for cls, vals in metrics.items():
        result_table["Class"].append(cls)
        result_table["Precision"].append(vals["Precision"])
        result_table["Recall"].append(vals["Recall"])
    
    # Now calculate overall precision and recall across all classes.
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    # Add the overall precision and recall to the result table.
    result_table["Class"].append("Overall")
    result_table["Precision"].append(overall_precision)
    result_table["Recall"].append(overall_recall)
    
    # Return the metrics as a pandas DataFrame, which can be displayed in the Gradio interface.
    return pd.DataFrame(result_table)

# Gradio function to handle image uploads, perform predictions, and return the results.
def predict_image(input_image_path):
    # Open the uploaded image and convert it to RGB format (removes any alpha channels).
    image = Image.open(input_image_path).convert("RGB")
    
    # Perform predictions on the image and get the result image and list of predictions.
    result_image, predictions = predict(model, image)
    
    # Calculate precision and recall based on the predictions.
    metrics_table = calculate_metrics(predictions)
    
    # Return the processed image with bounding boxes and the metrics table.
    return result_image, metrics_table

# Set up the Gradio interface for the web app.
interface = gr.Interface(
    fn=predict_image,  # The function to run when an image is uploaded.
    inputs=gr.Image(type="filepath"),  # Input widget for image uploads.
    outputs=[
        gr.Image(type="pil"),  # Output widget for displaying the image with bounding boxes.
        gr.Dataframe(headers=["Class", "Precision", "Recall"], type="pandas"),  # Output widget for showing precision/recall metrics.
    ],
    title="Blood Cell Detection App",  # Title for the app.
    description="Upload an image of blood cells to detect RBCs, WBCs, and Platelets."  # Short description of the app.
)

# Launch the Gradio web interface to start the app.
interface.launch()
