import os
import cv2
import xml.etree.ElementTree as ET
import albumentations as A

# Define paths to input image and XML directories, as well as the output directories for augmented images and XMLs.
input_images_path = r'C:\Users\AYUSH\OneDrive - Vidyalankar Institute of Technology\Desktop\ML_INTERN_01\BCCD_Dataset\BCCD\JPEGImages'
input_xml_path = r'C:\Users\AYUSH\OneDrive - Vidyalankar Institute of Technology\Desktop\ML_INTERN_01\BCCD_Dataset\BCCD\Annotations'
output_images_path = r'C:\Users\AYUSH\OneDrive - Vidyalankar Institute of Technology\Desktop\ML_INTERN_01\BCCD_Dataset\BCCD\Augmented_images'
output_xml_path = r'C:\Users\AYUSH\OneDrive - Vidyalankar Institute of Technology\Desktop\ML_INTERN_01\BCCD_Dataset\BCCD\Augmented_xml'

# Define augmentation transformations using the Albumentations library.
# These include random cropping, horizontal flipping, rotation, brightness/contrast adjustments, and Gaussian blur.
transform = A.Compose([
    A.RandomCrop(width=416, height=416, p=0.5),   # Crop random part of the image with 50% probability.
    A.HorizontalFlip(p=0.5),                      # Flip image horizontally with 50% probability.
    A.Rotate(limit=90, p=0.5),                    # Randomly rotate the image within a range of -90 to 90 degrees.
    A.RandomBrightnessContrast(p=0.5),            # Randomly adjust brightness and contrast with 50% probability.
    A.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.5),  # Apply Gaussian blur with a random sigma value.
])

# Function to parse the XML annotation file and extract bounding box information (for each object).
# The bounding box coordinates and the object names (e.g., RBC, WBC) are returned.
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        name = obj.find('name').text  # Extract class name (e.g., RBC, WBC, Platelets).
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)  # Extract xmin coordinate of the bounding box.
        ymin = int(bndbox.find('ymin').text)  # Extract ymin coordinate of the bounding box.
        xmax = int(bndbox.find('xmax').text)  # Extract xmax coordinate of the bounding box.
        ymax = int(bndbox.find('ymax').text)  # Extract ymax coordinate of the bounding box.
        boxes.append((name, xmin, ymin, xmax, ymax))  # Store object name and bounding box as a tuple.
    return boxes

# Function to save the augmented XML annotation by modifying the original XML file.
# The updated bounding boxes are used to replace the original ones, and the new XML is saved.
def save_augmented_xml(original_xml, new_boxes, output_xml_path):
    tree = ET.parse(original_xml)  # Parse the original XML.
    root = tree.getroot()

    # Remove all 'object' tags from the XML since we will add the new augmented boxes.
    for obj in root.iter('object'):
        root.remove(obj)

    # Create new 'object' entries for each augmented bounding box and add them to the XML.
    for box in new_boxes:
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = box[0]  # Add the object name (e.g., RBC, WBC).
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(box[1])  # Add xmin coordinate.
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(box[2])  # Add ymin coordinate.
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(box[3])  # Add xmax coordinate.
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(box[4])  # Add ymax coordinate.

    # Save the modified XML file in the output directory.
    new_xml_file = os.path.join(output_xml_path, os.path.basename(original_xml))
    tree.write(new_xml_file)

# Limit the total number of images for augmentation.
max_images = 1000  # Augment up to 1000 images.
image_count = 0  # Initialize counter for how many images have been processed.

# Loop over all images in the input directory.
for filename in os.listdir(input_images_path):
    if filename.endswith('.jpg'):  # Ensure we are only working with .jpg files (images).
        if image_count >= max_images:  # Stop if we have reached the maximum number of augmented images.
            break

        # Construct the full paths for the image and its corresponding XML annotation file.
        img_path = os.path.join(input_images_path, filename)
        xml_path = os.path.join(input_xml_path, filename.replace('.jpg', '.xml'))

        # Load the image using OpenCV.
        image = cv2.imread(img_path)

        # Parse the XML to get the bounding box coordinates and class labels for the objects in the image.
        boxes = parse_xml(xml_path)

        # Apply the defined augmentation transformations to the image.
        augmented = transform(image=image)
        augmented_image = augmented['image']

        # Save the augmented image to the output directory with a new file name.
        new_image_file = os.path.join(output_images_path, f'aug_{image_count + 1}.jpg')
        cv2.imwrite(new_image_file, augmented_image)

        # Prepare the new bounding boxes for saving in the augmented XML file.
        new_boxes = []
        for box in boxes:
            name, xmin, ymin, xmax, ymax = box
            # In this version, we are not modifying the bounding boxes themselves (just using the original ones).
            new_boxes.append((name, xmin, ymin, xmax, ymax))

        # Save the augmented XML annotations in the output XML directory.
        save_augmented_xml(xml_path, new_boxes, output_xml_path)

        # Increment the counter after each augmented image and XML pair is saved.
        image_count += 1

# After all images are processed, print a completion message.
print("Augmentation complete!")
