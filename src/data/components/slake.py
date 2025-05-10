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


def collate_fn(batch):
    return {
        'image': torch.stack([item['image'] for item in batch]),
        # 'seg_mask': torch.stack([item['seg_mask'] for item in batch]),
        'detection_objects': [item['detection_objects'] for item in batch],
        'detection_boxes': [item['detection_boxes'] for item in batch],
        'question': [item['question'] for item in batch],
        'answer': [item['answer'] for item in batch],
        'caption': [item['caption'] for item in batch], # concatenation of quesiton and answer 
        'modality': [item['modality'] for item in batch], # MRI, CT, 
        'abnormal': torch.tensor([item['abnormal'] for item in batch]),
        'organ': [item['organ'] for item in batch], 
        'label': torch.tensor([item['label'] for item in batch]),
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

class SlakeDataset(Dataset):
    def __init__(self, 
            dataset_root_path, 
            split='train', 
            mask_transform=None,
            transform=None,
            content_type: str=None, 
            modality: str=None,
            language: str="en", 
            img_id_limit: int=-1,
            label_type: str="abnormal",
        ):
        self.split = split
        self.ann = []
        self.ann_file = os.path.join(dataset_root_path, split + '.json')
        self.ann += json.load(open(self.ann_file, 'r'))

        # filter out samples 
        if content_type and content_type != "":
            self.ann = [item for item in self.ann if item['content_type'] == content_type]
        if modality and modality != "":
            self.ann = [item for item in self.ann if item['modality'] == modality]
        if language and language != "":
            self.ann = [item for item in self.ann if item['q_lang'] == language]
        if img_id_limit and img_id_limit > 0:
            self.ann = [item for item in self.ann if item['img_id'] <= img_id_limit]
            
        if mask_transform is not None:
            self.mask_transform = mask_transform
        else:
            self.mask_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor()
            ])
        self.transform = transform
        self.img_root = os.path.join(dataset_root_path, 'imgs')
        self.label_type = label_type

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.img_root, ann['img_name'])
        img_base_path = '/'.join(image_path.split('/')[:-1])
        image = np.array(Image.open(image_path))
        if self.transform:
            image = self.transform(image)
        
        seg_mask_path = os.path.join(img_base_path, 'mask.png')
        seg_mask = Image.open(seg_mask_path)
        if self.mask_transform:
            seg_mask = self.mask_transform(seg_mask)

        question = ann['question']
        answers = ann['answer']
        organ = ann['location']
        imaging_map = {"MRI": 0, "CT": 1, "X-Ray": 2}
        modality = imaging_map.get(ann["modality"], -1) # need to convert to numerical 
        
        # define classification labels
        if ann['content_type'].lower() == "abnormality":
            if "healthy" in question or "normal" in question:
                abnormal = 0 if answers.lower() == "yes" else 1 
            elif "abnormalit" in question:
                abnormal = 1 if answers.lower() == "yes" else 0 
            elif "disease" in question: 
                abnormal = 1
            else:
                #print(f"WARNING: Not categorized as healthy or abnormal. Check {question}, {answers}")
                abnormal = 1 if answers.lower() == "yes" else 0 
        else:
            abnormal = 0 # check if this assignment is okay 
   
        with open(os.path.join(img_base_path, 'detection.json'), 'r') as f:
            detection_dict = json.load(f)
            detection_objects = [list(ele.keys())[0] for ele in detection_dict]
            detection_boxes = [list(ele.values())[0] for ele in detection_dict]

        if "abnormal" in self.label_type.lower(): label = abnormal
        elif "organ" in self.label_type.lower(): label = organ 
        elif "modality" in self.label_type.lower(): label = modality 
        else: label = abnormal
            
        return {
            'image': image,
            'seg_mask': seg_mask,
            'detection_objects': detection_objects,
            'detection_boxes': detection_boxes,
            'question': question,
            'caption': question + answers,
            'answer': answers,
            'modality': modality, # MRI, CT, 
            'abnormal': abnormal,
            'organ': organ, # @TODO need to add code to handle the multi-class string labels
            'label': label,
        }