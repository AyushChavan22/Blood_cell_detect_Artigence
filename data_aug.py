import os
from xml.etree import ElementTree as ET
from PIL import Image
import albumentations as A
import cv2

# Function to parse the Pascal VOC XML annotation
def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    labels = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    
    return boxes, labels

# Function to save the augmented annotations in Pascal VOC format
def save_voc_annotation(xml_file, boxes, labels, xml_save_path, augmented_image_size):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Update image size
    size_elem = root.find('size')
    width_elem = size_elem.find('width')
    height_elem = size_elem.find('height')
    width_elem.text = str(augmented_image_size[1])
    height_elem.text = str(augmented_image_size[0])

    # Remove existing object elements
    for obj in root.findall('object'):
        root.remove(obj)
    
    # Append new objects with updated bounding boxes and labels
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        obj_elem = ET.Element('object')
        
        name_elem = ET.Element('name')
        name_elem.text = label
        obj_elem.append(name_elem)
        
        bbox_elem = ET.Element('bndbox')
        xmin_elem = ET.Element('xmin')
        ymin_elem = ET.Element('ymin')
        xmax_elem = ET.Element('xmax')
        ymax_elem = ET.Element('ymax')
        
        xmin_elem.text = str(int(xmin))
        ymin_elem.text = str(int(ymin))
        xmax_elem.text = str(int(xmax))
        ymax_elem.text = str(int(ymax))
        
        bbox_elem.append(xmin_elem)
        bbox_elem.append(ymin_elem)
        bbox_elem.append(xmax_elem)
        bbox_elem.append(ymax_elem)
        obj_elem.append(bbox_elem)
        
        root.append(obj_elem)
    
    # Save the updated XML file in the XML save directory
    xml_save_file = os.path.join(xml_save_path, "augmented_" + os.path.basename(xml_file))
    tree.write(xml_save_file)

# Function to apply augmentation and save the augmented images
def apply_augmentation(image_path, xml_path, image_save_path, xml_save_path):
    # Read the image
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]
    
    # Adjust crop size dynamically based on the image dimensions
    crop_height = min(512, img_height)
    crop_width = min(512, img_width)
    
    # Define the augmentation pipeline
    transform = A.Compose([
        A.ColorJitter(p=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        A.Rotate(limit=40, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomScale(scale_limit=(-0.2, 0.5), p=0.5),
        A.RandomCrop(width=crop_width, height=crop_height, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ], bbox_params=A.BboxParams(format='pascal_voc',  min_area=500, min_visibility=0.1, label_fields=['labels']))

    # Parse the bounding boxes and labels from the corresponding XML
    boxes, labels = parse_voc_annotation(xml_path)

    # Convert bounding boxes to the Albumentations format
    bboxes = [[box[0], box[1], box[2], box[3]] for box in boxes]

    # Apply augmentation (including bounding box adjustment)
    augmented = transform(image=image, bboxes=bboxes, labels=labels)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_labels = augmented['labels']

    # Convert bounding boxes back to Pascal VOC format (xmin, ymin, xmax, ymax)
    augmented_boxes = [[box[0], box[1], box[2], box[3]] for box in augmented_bboxes]

    # Save the augmented image
    image_name = os.path.basename(image_path)
    augmented_image_path = os.path.join(image_save_path, f"augmented_{image_name}")
    augmented_image_size = augmented_image.shape[:2]

    # Save augmented image in the separate directory
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    cv2.imwrite(augmented_image_path, augmented_image)

    # Save the augmented bounding boxes and labels as Pascal VOC annotations in a different directory
    save_voc_annotation(xml_path, augmented_boxes, augmented_labels, xml_save_path, augmented_image_size)

def apply_augmentation_dir(image_dir, xml_dir, image_save_dir, xml_save_dir):
    # Get the list of images and annotations
    image_list = os.listdir(image_dir)
    xml_list = os.listdir(xml_dir)

    # Sort the lists to ensure the order of images and annotations is the same
    image_list.sort()
    xml_list.sort()

    # Apply augmentation to each image and annotation
    for image_name, xml_name in zip(image_list, xml_list):
        image_path = os.path.join(image_dir, image_name)
        xml_path = os.path.join(xml_dir, xml_name)
        try:
            apply_augmentation(image_path, xml_path, image_save_dir, xml_save_dir)
            print(f"Augmented {image_name}")
        except Exception as e:
            print(e)
            print(f"Error in processing {image_name}")

# Example usage:
if __name__ == "__main__":
    # Directory containing original images and annotations
    image_dir = r"path to your images"
    xml_dir = r"path to your annotations file"
    
    # Directories to save augmented images and XML annotations separately
    image_save_dir = r"path to save augmented images"
    xml_save_dir = r"path to save augmented xml"

    # Apply augmentations and save results
    apply_augmentation_dir(image_dir, xml_dir, image_save_dir, xml_save_dir)
