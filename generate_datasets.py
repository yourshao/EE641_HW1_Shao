#!/usr/bin/env python3
"""
Generate synthetic datasets for EE641 HW1.
This script creates both the shape detection and keypoint datasets.

Usage:
    python generate_datasets.py --output_dir ./datasets
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
import argparse

def generate_shape_detection_dataset(output_dir, num_train=1000, num_val=200):
    """Generate synthetic shape detection dataset."""
    
    # Create directory structure
    train_dir = os.path.join(output_dir, 'detection', 'train')
    val_dir = os.path.join(output_dir, 'detection', 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Class definitions
    classes = [
        {'id': 0, 'name': 'circle', 'size_range': (16, 32)},
        {'id': 1, 'name': 'square', 'size_range': (48, 96)},
        {'id': 2, 'name': 'triangle', 'size_range': (96, 192)}
    ]
    
    def create_shape_image(image_id, num_objects, split='train'):
        """Create a single image with random shapes."""
        img_size = 224
        img = Image.new('RGB', (img_size, img_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        annotations = []
        occupied_regions = []
        
        for _ in range(num_objects):
            # Random class
            cls = random.choice(classes)
            cls_id = cls['id']
            
            # Random size within class range
            size = random.randint(*cls['size_range'])
            
            # Try to find non-overlapping position
            max_attempts = 50
            for attempt in range(max_attempts):
                x = random.randint(size//2, img_size - size//2)
                y = random.randint(size//2, img_size - size//2)
                
                # Check overlap with existing objects
                bbox = [x - size//2, y - size//2, x + size//2, y + size//2]
                overlap = False
                for existing in occupied_regions:
                    if compute_iou_simple(bbox, existing) > 0.3:
                        overlap = True
                        break
                
                if not overlap or attempt == max_attempts - 1:
                    # Draw shape
                    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][cls_id]
                    
                    if cls['name'] == 'circle':
                        draw.ellipse(bbox, fill=color, outline='black', width=2)
                    elif cls['name'] == 'square':
                        draw.rectangle(bbox, fill=color, outline='black', width=2)
                    elif cls['name'] == 'triangle':
                        points = [
                            (x, y - size//2),  # top
                            (x - size//2, y + size//2),  # bottom left
                            (x + size//2, y + size//2)   # bottom right
                        ]
                        draw.polygon(points, fill=color, outline='black', width=2)
                        # Update bbox for triangle
                        bbox = [x - size//2, y - size//2, x + size//2, y + size//2]
                    
                    occupied_regions.append(bbox)
                    annotations.append({
                        'bbox': bbox,
                        'category_id': cls_id,
                        'area': size * size
                    })
                    break
        
        # Save image
        save_dir = train_dir if split == 'train' else val_dir
        img_path = os.path.join(save_dir, f'{image_id:06d}.png')
        img.save(img_path)
        
        return {
            'id': image_id,
            'file_name': f'{image_id:06d}.png',
            'width': img_size,
            'height': img_size,
            'annotations': annotations
        }
    
    def compute_iou_simple(box1, box2):
        """Simple IoU computation for overlap checking."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Generate training set
    print("Generating shape detection training set...")
    train_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': c['id'], 'name': c['name']} for c in classes]
    }
    
    annotation_id = 0
    for i in range(num_train):
        num_objects = random.randint(2, 5)
        img_info = create_shape_image(i, num_objects, 'train')
        train_data['images'].append({
            'id': img_info['id'],
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height']
        })
        
        for ann in img_info['annotations']:
            train_data['annotations'].append({
                'id': annotation_id,
                'image_id': img_info['id'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area']
            })
            annotation_id += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_train} training images")
    
    # Generate validation set
    print("Generating shape detection validation set...")
    val_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': c['id'], 'name': c['name']} for c in classes]
    }
    
    for i in range(num_val):
        num_objects = random.randint(2, 5)
        img_info = create_shape_image(num_train + i, num_objects, 'val')
        val_data['images'].append({
            'id': img_info['id'],
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height']
        })
        
        for ann in img_info['annotations']:
            val_data['annotations'].append({
                'id': annotation_id,
                'image_id': img_info['id'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area']
            })
            annotation_id += 1
    
    # Save annotations
    with open(os.path.join(output_dir, 'detection', 'train_annotations.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'detection', 'val_annotations.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Shape detection dataset generated: {num_train} train, {num_val} val images")


def generate_keypoint_dataset(output_dir, num_train=1000, num_val=200):
    """Generate synthetic stick figure keypoint dataset."""
    
    # Create directory structure
    train_dir = os.path.join(output_dir, 'keypoints', 'train')
    val_dir = os.path.join(output_dir, 'keypoints', 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Keypoint definitions
    keypoint_names = ['head', 'left_hand', 'right_hand', 'left_foot', 'right_foot']
    
    def create_stick_figure(image_id, split='train'):
        """Create a single stick figure image."""
        img_size = 128
        img = Image.new('L', (img_size, img_size), color=255)  # Grayscale
        draw = ImageDraw.Draw(img)
        
        # Random position and pose
        center_x = random.randint(40, 88)
        center_y = random.randint(40, 88)
        
        # Define stick figure proportions
        head_radius = 8
        torso_length = 25
        arm_length = 20
        leg_length = 25
        
        # Random pose variations
        arm_angle_left = random.uniform(-60, 60)  # degrees
        arm_angle_right = random.uniform(-60, 60)
        leg_spread = random.uniform(10, 30)
        
        # Calculate keypoint positions
        head_pos = (center_x, center_y - torso_length//2 - head_radius)
        torso_top = (center_x, center_y - torso_length//2)
        torso_bottom = (center_x, center_y + torso_length//2)
        
        # Arms
        left_hand = (
            center_x - int(arm_length * np.cos(np.radians(90 - arm_angle_left))),
            center_y - torso_length//4 + int(arm_length * np.sin(np.radians(90 - arm_angle_left)))
        )
        right_hand = (
            center_x + int(arm_length * np.cos(np.radians(90 - arm_angle_right))),
            center_y - torso_length//4 + int(arm_length * np.sin(np.radians(90 - arm_angle_right)))
        )
        
        # Legs
        left_foot = (
            center_x - int(leg_spread),
            center_y + torso_length//2 + leg_length
        )
        right_foot = (
            center_x + int(leg_spread),
            center_y + torso_length//2 + leg_length
        )
        
        # Draw stick figure
        line_width = 3
        
        # Head
        draw.ellipse([head_pos[0] - head_radius, head_pos[1] - head_radius,
                     head_pos[0] + head_radius, head_pos[1] + head_radius],
                    fill=None, outline=0, width=line_width)
        
        # Torso
        draw.line([torso_top, torso_bottom], fill=0, width=line_width)
        
        # Arms
        draw.line([(center_x, center_y - torso_length//4), left_hand], fill=0, width=line_width)
        draw.line([(center_x, center_y - torso_length//4), right_hand], fill=0, width=line_width)
        
        # Legs
        draw.line([torso_bottom, left_foot], fill=0, width=line_width)
        draw.line([torso_bottom, right_foot], fill=0, width=line_width)
        
        # Save image
        save_dir = train_dir if split == 'train' else val_dir
        img_path = os.path.join(save_dir, f'{image_id:06d}.png')
        img.save(img_path)
        
        # Keypoints in order: head, left_hand, right_hand, left_foot, right_foot
        keypoints = [head_pos, left_hand, right_hand, left_foot, right_foot]
        
        return {
            'id': image_id,
            'file_name': f'{image_id:06d}.png',
            'keypoints': keypoints
        }
    
    # Generate training set
    print("Generating keypoint training set...")
    train_annotations = []
    for i in range(num_train):
        ann = create_stick_figure(i, 'train')
        train_annotations.append(ann)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_train} training images")
    
    # Generate validation set
    print("Generating keypoint validation set...")
    val_annotations = []
    for i in range(num_val):
        ann = create_stick_figure(num_train + i, 'val')
        val_annotations.append(ann)
    
    # Save annotations
    train_data = {
        'images': train_annotations,
        'keypoint_names': keypoint_names,
        'num_keypoints': len(keypoint_names)
    }
    
    val_data = {
        'images': val_annotations,
        'keypoint_names': keypoint_names,
        'num_keypoints': len(keypoint_names)
    }
    
    with open(os.path.join(output_dir, 'keypoints', 'train_annotations.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'keypoints', 'val_annotations.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Keypoint dataset generated: {num_train} train, {num_val} val images")



def main():
    parser = argparse.ArgumentParser(description='Generate datasets for EE641 HW1')
    parser.add_argument('--output_dir', type=str, default='./datasets',
                       help='Output directory for datasets')
    parser.add_argument('--num_train', type=int, default=1000,
                       help='Number of training images per dataset')
    parser.add_argument('--num_val', type=int, default=200,
                       help='Number of validation images per dataset')
    parser.add_argument('--seed', type=int, default=641,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Generate datasets
    print("="*50)
    print("EE641 HW1 Dataset Generation")
    print(f"Seed: {args.seed}")
    print("="*50)
    
    generate_shape_detection_dataset(args.output_dir, args.num_train, args.num_val)
    print()
    generate_keypoint_dataset(args.output_dir, args.num_train, args.num_val)
    
    print("\nDataset generation complete!")
    print(f"Location: {args.output_dir}/")


if __name__ == '__main__':
    main()