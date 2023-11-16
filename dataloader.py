import argparse
import os
import glob
import random
import collections
import pandas as pd
import numpy as np
import webdataset as wds
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from imwatermark import WatermarkEncoder
from utils_sampling import UnderSampler
from PIL import ImageFile, Image


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000 

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

    train_transform = v2.Compose([
        v2.Lambda(crop_bottom),
        v2.RandomCrop((256, 256), pad_if_needed=True),
        v2.Lambda(random_gaussian_blur),
        v2.RandomGrayscale(p=0.05),
        v2.ToPILImage(),
        v2.Lambda(random_invisible_watermark),
        v2.ToImage()
    ])

    eval_transform = v2.Compose([
        v2.CenterCrop((256, 256)),
    ])

    transform = train_transform if split == "train" else eval_transform

    def identity(x):
        return x

    dataset = wds.DataPipeline(
        wds.SimpleShardList(all_files),
        wds.shuffle(100),
        wds.tarfile_to_samples(),
        wds.shuffle(3000, initial=3000),
        wds.decode("torchrgb"),
        wds.to_tuple("jpg", "label.cls", "domain_label.cls"),
        wds.map_tuple(transform, identity, identity),
        wds.batched(8),
        wds.shuffle(1000, initial=500),
        wds.unbatched(),
    )

    return dataset


def load_dataloader(domains: list[int], split: str, batch_size: int = 32, num_workers: int = 8):
    dataset = load_dataset(domains, split)

    if split == "train":
        sampler = UnderSampler(dataset, {0: 0.5, 1: 0.5}, seed=42)
        dataloader = DataLoader(sampler, batch_size=batch_size, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return dataloader

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    # testing code
    dl = load_dataloader([1, 4], "train")
    y_dist = collections.Counter()
    d_dist = collections.Counter()

    for i, (img, y, d) in tqdm(enumerate(dl)):
        if i > 197:
            print(y, d)
        if i == 200:
            break
        y_dist.update(y.numpy())
        d_dist.update(d.numpy())
    
    print("class label")
    for label in sorted(y_dist):
        frequency = y_dist[label] / sum(y_dist.values())
        print(f'• {label}: {frequency:.2%} ({y_dist[label]})')
    
    print("domain label")
    for label in sorted(d_dist):
        frequency = d_dist[label] / sum(d_dist.values())
        print(f'• {label}: {frequency:.2%} ({d_dist[label]})')