import os
import xml.etree.ElementTree as ET

# Paths for input and output data
# 'input_xml_path' is where the original XML annotation files are stored, which we'll convert to YOLO format.
# 'output_yolo_path' is the folder where the converted YOLO text files will be saved.
input_xml_path = r'C:\Users\AYUSH\OneDrive - Vidyalankar Institute of Technology\Desktop\ML_INTERN_01\BCCD_Dataset\BCCD\Annotations'
output_yolo_path = r'C:\Users\AYUSH\OneDrive - Vidyalankar Institute of Technology\Desktop\ML_INTERN_01\BCCD_Dataset\BCCD\Augmented_yolo'

# If the output directory does not exist, this will create it to ensure that the YOLO text files can be saved properly.
os.makedirs(output_yolo_path, exist_ok=True)

# Mapping the class labels from the dataset to numerical labels used in YOLO format.
# In this case, 'RBC' is labeled as 0, 'WBC' as 1, and 'Platelet' as 2.
class_mapping = {
    'RBC': 0,
    'WBC': 1,
    'Platelet': 2
}

# This function handles the conversion of each XML file to YOLO format.
# It takes the bounding boxes from the XML file, calculates the center of each box, 
# and normalizes the coordinates based on image dimensions to create a YOLO-friendly format.
def convert_xml_to_yolo(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Extract the image width and height from the XML file. This information is crucial 
    # to normalize the bounding box coordinates to values between 0 and 1, which YOLO requires.
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)
    
    yolo_boxes = []
    
    # Loop through all the 'object' tags in the XML file. These contain information about the detected objects.
    for obj in root.iter('object'):
        # Get the class name of the object (e.g., 'RBC', 'WBC', 'Platelet').
        name = obj.find('name').text
        
        # Check if the class name exists in our mapping. If it does, get the corresponding class ID (0, 1, or 2).
        if name in class_mapping:
            class_id = class_mapping[name]
            
            # Extract the bounding box coordinates (xmin, ymin, xmax, ymax) from the XML file.
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Convert the bounding box coordinates to YOLO format:
            # YOLO format requires the center coordinates of the box and its width and height, all normalized.
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            # Create a string with the class ID and the normalized bounding box values,
            # and append it to the list of YOLO-format boxes.
            yolo_boxes.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return yolo_boxes


# Iterate through all XML files in the input directory and convert each one to YOLO format.
for filename in os.listdir(input_xml_path):
    # Check if the current file is an XML file.
    if filename.endswith('.xml'):
        # Get the full path to the XML file.
        xml_path = os.path.join(input_xml_path, filename)
        
        # Convert the XML file to YOLO format by calling the function defined earlier.
        yolo_boxes = convert_xml_to_yolo(xml_path)
        
        # Now we need to save the YOLO format data to a text file with the same name as the XML file,
        # but with a '.txt' extension instead of '.xml'. This text file will go into the output directory.
        yolo_file_path = os.path.join(output_yolo_path, filename.replace('.xml', '.txt'))
        
        # Open the new file and write all the YOLO-format bounding boxes into it.
        with open(yolo_file_path, 'w') as yolo_file:
            yolo_file.write("\n".join(yolo_boxes))

# Print a message to indicate that the conversion is complete.
print("Conversion to YOLO format complete!")
