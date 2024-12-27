import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_paths = []
        self.labels = []

        split_path = os.path.join(root_dir, f"{split}.csv")
        with open(split_path, "r") as f:
            next(f)
            for line in f:
                img_name, label = line.strip().split(",")
                self.image_paths.append(os.path.join(root_dir, "images", img_name))
                self.labels.append(label)
        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        label = self.label2idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label


def init_dataset(data_src_path) -> MiniImageNetDataset:
    # Define the dataset
    dataset = MiniImageNetDataset(
        root_dir=data_src_path,
        split="train",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    return dataset



# test_core.py
import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader_param_helper.core import find_optimal_params

class TestCoreFunctions(unittest.TestCase):
    def test_cuda_batchsize_128(self):
        data_set = init_dataset("../dataset/mini-imagenet")
        result = find_optimal_params(data_set, torch.device("cuda"), 128)
        print(result)

    def test_cpu_batchsize_64(self):
        data_set = init_dataset("../dataset/mini-imagenet")
        result = find_optimal_params(data_set, torch.device("cpu"), 64)
        print(result)

if __name__ == '__main__':
    #unittest.main()
    
    if True:
        device_list = [torch.device("cuda"), torch.device("cpu")]
        batch_size_lst = [8, 32]
        data_set = init_dataset("../dataset/mini-imagenet")
        
        for device in device_list:
            for batch_size in batch_size_lst:
                
                result_dict = find_optimal_params(data_set, device, batch_size)
                print(f">>> Result in {device.type}, {batch_size} batch -> {result_dict}\n")
    
