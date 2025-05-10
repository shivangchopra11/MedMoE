import os
import re
import math
import json
import torch
import random
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
from src.data.data_utils import get_data

def collate_fn(batch):
    return {
        'image': [x[0] for x in batch],
        'caption': [x[1] for x in batch],
        'label': torch.tensor([x[2] for x in batch], dtype=torch.long),
    }


class ImageMaskingGenerator:
    def __init__(
        self,
        input_size: Union[Tuple[int, int], int],
        num_masking_patches: int,
        min_num_patches: int = 4,
        max_num_patches: Optional[int] = None,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
    ) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self) -> str:
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self) -> Tuple[int, int]:
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        delta = 0
        for _attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self) -> np.ndarray:
        mask = np.zeros(shape=self.get_shape(), dtype=np.int64)  # type: ignore
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

class UnimedDataset(Dataset):
    def __init__(self, dataset_root_path, 
        jsonl_file='combined_data.jsonl',
        split='train', 
        transform=None,
    ):

        self.transform=transform
        train_data = "/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/radimagenet_webdataset/dataset-{000001..001049}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/chexpert_webdataset/dataset-{000001..000212}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/openi_webdataset/dataset-{000001..000007}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/chest_xray8_webdataset/dataset-{000001..000113}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/mimic_cxr/dataset-{000001..000270}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/roco_webdataset/dataset-{000001..000061}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/pmc_clip_webdataset/dataset-{000001..001645}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/llava_med_alignment_set_webdataset/dataset-{000001..000468}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/llava_med_hq_60k_set_webdataset/dataset-{000001..000265}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/quilt_webdataset/dataset-{000001..001018}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/retina_part1_webdataset/dataset-{000001..000155}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/retina_part2_webdataset/dataset-{000001..000013}.tar::/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/retina_part3_webdataset/dataset-{000001..000006}.tar"
        self.data = get_data(train_data, None, self.transform, dataset_type='auto')['train']
        # self.split = split
        # self.ann = []
        
        # print("dataset_root_path:",dataset_root_path)  # Debugging line
        # self.ann_file = os.path.join(dataset_root_path, jsonl_file)
        # print("Annotation file path:", self.ann_file)  # Debugging line
        # with open(self.ann_file, 'r') as f:
        #     for line in f:
        #         self.ann.append(json.loads(line.strip()))

        # if split == 'valid' or split == 'test':    
        #     idx = np.random.choice(range(len(self.ann)), size=10000, replace=False)
        #     self.ann = [self.ann[i] for i in idx]
                  
        # self.transform = transform
        
        # self.img_root = os.path.join(dataset_root_path)
        

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        image_name = ann['image']
        caption = ann['caption']
        # label = ann['label']
        label = 0

        image_path = os.path.join(self.img_root, ann['image'])
        img_base_path = '/'.join(image_path.split('/')[:-1])

        # Construct the full path to the image
        image_path = os.path.join(self.img_root, image_name)
        
        # Load the image
        image =  np.array(Image.open(image_path).convert('RGB'))
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
        return image, caption, label