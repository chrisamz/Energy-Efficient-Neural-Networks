# data_preprocessing.py

"""
Data Preprocessing Module for Energy-Efficient Neural Networks

This module contains functions for collecting, cleaning, normalizing, and preparing data for training energy-efficient neural networks.

Techniques Used:
- Data cleaning
- Normalization
- Data augmentation

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- albumentations (for data augmentation)

"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import albumentations as A
import cv2

class DataPreprocessing:
    def __init__(self, raw_data_dir='data/raw/', processed_data_dir='data/processed/'):
        """
        Initialize the DataPreprocessing class.
        
        :param raw_data_dir: str, directory containing raw data
        :param processed_data_dir: str, directory to save processed data
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir

    def load_data(self, filename):
        """
        Load data from a CSV file.
        
        :param filename: str, name of the CSV file
        :return: DataFrame, loaded data
        """
        filepath = os.path.join(self.raw_data_dir, filename)
        data = pd.read_csv(filepath)
        return data

    def clean_data(self, data):
        """
        Clean the data by removing null values and duplicates.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.dropna().drop_duplicates()
        return data

    def normalize_data(self, data):
        """
        Normalize the data using standard scaling.
        
        :param data: DataFrame, input data
        :return: DataFrame, normalized data
        """
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
        return pd.DataFrame(normalized_data, columns=data.columns)

    def augment_image(self, image):
        """
        Apply data augmentation to an image.
        
        :param image: ndarray, input image
        :return: ndarray, augmented image
        """
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        augmented_image = transform(image=image)['image']
        return augmented_image

    def augment_data(self, data):
        """
        Apply data augmentation to the dataset.
        
        :param data: DataFrame, input data
        :return: DataFrame, augmented data
        """
        augmented_data = []
        for _, row in data.iterrows():
            image_path = row['image_path']
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented_image = self.augment_image(image)
            row['image'] = augmented_image
            augmented_data.append(row)
        return pd.DataFrame(augmented_data)

    def save_data(self, data, filename):
        """
        Save the processed data to a CSV file.
        
        :param data: DataFrame, processed data
        :param filename: str, name of the output CSV file
        """
        os.makedirs(self.processed_data_dir, exist_ok=True)
        filepath = os.path.join(self.processed_data_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def preprocess(self, filename):
        """
        Execute the full preprocessing pipeline.
        
        :param filename: str, name of the raw data file
        """
        # Load data
        data = self.load_data(filename)
        
        # Clean data
        data = self.clean_data(data)
        
        # Normalize data
        data = self.normalize_data(data)
        
        # Apply data augmentation
        data = self.augment_data(data)
        
        # Save processed data
        self.save_data(data, 'processed_data.csv')
        print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocessing = DataPreprocessing(raw_data_dir='data/raw/', processed_data_dir='data/processed/')
    
    # Execute the preprocessing pipeline
    preprocessing.preprocess('raw_data.csv')
    print("Data preprocessing completed and data saved.")
