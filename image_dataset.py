import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import ast

class MultiLabelImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Transformation to apply on an image.
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Parse the CSV file's categories into lists and multi-hot vectors
        def parse_categories(cat_str):
            # ast.literal_eval safely evaluates the string to a list
            cats = ast.literal_eval(cat_str)
            # Convert each category from float to int
            return [int(x) for x in cats]
        self.df['category_list'] = self.df['categories'].apply(parse_categories)

        # Function to create a multi-hot vector from the category list
        def multi_hot_vector(categories, num_classes):
            vector = np.zeros(num_classes, dtype=np.float32)
            for cat in categories:
                # Assuming CSV classes are 1-indexed, convert to 0-indexed:
                index = cat - 1
                if 0 <= index < num_classes:
                    vector[index] = 1.0
            return vector
        self.num_classes = 290
        self.df['multi_hot'] = self.df['category_list'].apply(lambda x: multi_hot_vector(x, self.num_classes))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the image filename from the 'id' column
        img_name = f"{self.df.iloc[idx]['id']}.png"
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load the image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        
        # Apply the transformations if defined
        if self.transform:
            image = self.transform(image)
            
        # Convert the multi-hot label to a PyTorch tensor
        label = torch.tensor(self.df.iloc[idx]['multi_hot'], dtype=torch.float32)
        
        return image, label


''' Only Need to do below to generate a new dataset

# Running Data Pipeline:

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # can tweak size and ratio
    transforms.ToTensor(),          # scales pixel values to [0,1]
])

img_directory = "data/train"
csv_file_path = "data/train_classification_labels.csv"

dataset = MultiLabelImageDataset(csv_file=csv_file_path,
                                 img_dir=img_directory,
                                 transform=transform)

torch.save(dataset, 'dataset.pt')

'''

