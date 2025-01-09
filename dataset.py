from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
import json
import os
from collections import defaultdict
from model import _PLACEHOLDER_IMAGE, _PLACEHOLDER_INSTRUCT

class MultimodalDataset(Dataset):
    def __init__(self, texts, images):
        """
        Dataset for multimodal training.

        Args:
            texts (list): List of text inputs.
            images (list): List of image file paths or PIL images.
            labels (list): List of text labels for the inputs.
        """
        self.texts = texts
        self.images = images

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize text
        text = self.texts[idx]
        # Process image
        if isinstance(self.images[idx], str):
            image = Image.open(self.images[idx]).convert("RGB")
        else:
            image = Image.fromarray(np.array(self.images[idx]))

        # Resize the image to 224x224
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        # Convert the resized image to a numpy array
        image = np.array(image)

        # Tokenize label (optional for training with labels)
        return {
                "input_texts": text,  # Remove batch dim
                "input_images": image,
                "labels": text,
            }

def get_dataloader(dataset, batch_size=2, shuffle=True):
    """
    Create a DataLoader from the dataset.
    
    Args:
        dataset (PhiClipMultimodalDataset): The dataset to load
        batch_size (int, optional): Number of samples per batch. Defaults to 2.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    
    Returns:
        DataLoader: A DataLoader for the dataset
    """
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
    )
    
def generate_phase_one_dataloader(annotation_file: str, image_dir: str, batch_size: int = 2):
    captions = [
        "Can you create a description for this photo?",
        "Could you write a caption for this picture?",
        "Generate a suitable caption for this image.",
        "Can you suggest a caption for this photo?",
        "Create a short caption for this image.",
        "Could you come up with a description for this picture?",
        "Generate a relevant caption for this photo.",
        "Can you draft a caption to match this image?",
        "Write a creative caption for this picture.",
        "Provide a caption for this image."
        ]
    with open(annotation_file) as file:
        data = json.load(file)
    
    
    annotation_dict = defaultdict(list)
    for item in data["annotations"]:
        annotation_dict[item["image_id"]].append(item["caption"])
    output = {}
    
    for item in data["images"]:
        output[item["file_name"]] = random.choice(annotation_dict[item["id"]])
     
    list_of_images = []
    list_of_texts = []
    for key in output.keys():
        if os.path.exists(os.path.join(image_dir, key)):
            list_of_texts.append(f"{random.choice(captions)} {_PLACEHOLDER_IMAGE} {_PLACEHOLDER_INSTRUCT} " + output[key])
            list_of_images.append(os.path.join(image_dir, key)) 
    
    dataset = MultimodalDataset(
        texts = list_of_texts,
        images = list_of_images,
    )
    return get_dataloader(dataset,batch_size) 

def generate_phase_two_dataloader(directory: str, batch_size: int = 2):
    # Parse the JSON string
    with open(os.path.join(directory, "filter_cap.json")) as file:
        data = json.load(file)
    annotations = data['annotations']
    image_dir = os.path.join(directory, "image")
    list_of_images = []
    list_of_texts = []
    for item in annotations:
        if os.path.exists(os.path.join(image_dir, f"{item["image_id"]}.jpg")):
            list_of_texts.append(f"Describe the image {_PLACEHOLDER_IMAGE} {_PLACEHOLDER_INSTRUCT} " + item["caption"])
            list_of_images.append(os.path.join(image_dir, f"{item["image_id"]}.jpg"))
    dataset = MultimodalDataset(
        texts = list_of_texts,
        images = list_of_images,
    )
    return get_dataloader(dataset,batch_size)
