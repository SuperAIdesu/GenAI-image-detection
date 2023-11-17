import argparse
import os
import glob
import random
import collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchdata as td, torchdata.datapipes as dp
from typing import Iterator
from torchdata.datapipes.iter import FileLister, FileOpener, Concater
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from tqdm import tqdm
from torchvision.transforms import v2
from torch.utils.data import DataLoader, RandomSampler
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

N_SAMPLES = {
    0: (115346, 14418, 14419),
    1: (22060, 2757, 2758),
    4: (21096, 2637, 2637),
    2: (13582, 1697, 1699),
    3: (12027, 1503, 1504)
}


@dp.functional_datapipe("collect_from_workers")
class WorkerResultCollector(dp.iter.IterDataPipe):
    def __init__(self, source: dp.iter.IterDataPipe):
        self.source = source

    def __iter__(self) -> Iterator:
        yield from self.source

    def is_replicable(self) -> bool:
        """Method to force data back to main process"""
        return False


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


def build_transform(split: str):
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

    return transform


def dp_to_tuple_train(input_dict):
    transform = build_transform("train")
    return transform(input_dict[".jpg"]), input_dict[".label.cls"], input_dict[".domain_label.cls"]


def dp_to_tuple_eval(input_dict):
    transform = build_transform("eval")
    return transform(input_dict[".jpg"]), input_dict[".label.cls"], input_dict[".domain_label.cls"]


def load_dataset(domains: list[int], split: str):

    laion_lister = FileLister("./data/laion400m_data", f"{split}*.tar")
    domain_names = [DOMAIN_LABELS[domain] for domain in domains]
    genai_listers = [FileLister(f"./data/genai-images/{domain}", f"{split}*.tar") for domain in domain_names]
    
    all_listers = Concater(laion_lister, *genai_listers).shuffle().sharding_filter()

    def open_lister(lister):
        opener = FileOpener(lister, mode="b")
        return opener.load_from_tar().routed_decode().webdataset()
    
    dp = open_lister(all_listers)
    
    if split == "train":
        dp = dp.map(dp_to_tuple_train)
    else:
        dp = dp.map(dp_to_tuple_eval)
    
    dp = dp.shuffle(buffer_size=3000).batch(8).collect_from_workers().shuffle(buffer_size=1024, unbatch_level=1)

    return dp


def load_dataloader(domains: list[int], split: str, batch_size: int = 32, num_workers: int = 4):
    dp = load_dataset(domains, split)
    # dp = dp.batch(batch_size).collate()
    # rs = MultiProcessingReadingService(num_workers=num_workers)
    # dl = DataLoader2(dp, reading_service=rs)
    sampler = UnderSampler(dp, {0: 0.5, 1: 0.5}, seed=42)
    dl = DataLoader(sampler, batch_size=batch_size, num_workers=num_workers)

    return dl

    
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