import argparse
import os
import glob
import random
import pandas as pd
import numpy as np
import webdataset as wds
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from imwatermark import WatermarkEncoder


encoder = WatermarkEncoder()
encoder.set_watermark('bytes', 'test'.encode('utf-8'))


DOMAIN_LABELS = {
    0: "laion",
    1: "StableDiffusion",
    2: "dalle2",
    3: "dalle3",
    4: "midjourney"
}


def crop_bottom(image, cutoff=16):
    return image[:, :-cutoff, :]


def random_gaussian_blur(image, p=0.01):
    if random.random() < p:
        return v2.functional.gaussian_blur(image, kernel_size=5)
    return image

def random_invisible_watermark(image, p=0.2):
    image = np.array(image)
    if random.random() < p:
        return encoder.encode(image, method='dwtDct')
    return image


def load_dataset(domains: list[int], split: str):
    laion_path = f"./data/laion400m_data/{split}*.tar"
    domain_names = [DOMAIN_LABELS[domain] for domain in domains]
    genai_paths = [f"./data/genai-images/{domain}/{split}*.tar" for domain in domain_names]
    combined_paths = [laion_path] + genai_paths
    all_files = [f for path in combined_paths for f in glob.glob(path)]
    dataset = wds.WebDataset(all_files).decode("torchrgb").to_tuple("jpg", "label.cls", "domain_label.cls")

    train_transform = v2.Compose([
        v2.Lambda(crop_bottom),
        v2.RandomCrop((256, 256)),
        v2.Lambda(random_gaussian_blur),
        v2.RandomGrayscale(p=0.05),
        v2.ToPILImage(),
        v2.Lambda(random_invisible_watermark),
        v2.ToImage()
    ])

    eval_transform = v2.Compose([
        v2.CenterCrop((256, 256)),
    ])

    def identity(x):
        return x

    if split == "train":
        dataset = dataset.map_tuple(train_transform, identity, identity)
    else:
        dataset = dataset.map_tuple(eval_transform, identity, identity)

    return dataset


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    load_dataset([1, 4], "val")