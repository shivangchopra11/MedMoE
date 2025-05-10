import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
# from albumentations import ShiftScaleRotate, Normalize, Resize, Compose
# from albumentations.pytorch import ToTensor
from datasets.download_chexpert import *


class ImageBaseDataset(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
    ):

        self.transform = transform
        self.split = split
        self.imsize = 256

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_jpg(self, img_path):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.imsize)
        img = Image.fromarray(x).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img

    def read_from_dicom(self, img_path):
        raise NotImplementedError

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img


class CheXpertImageDataset(ImageBaseDataset):
    def __init__(self, data_dir, sample_frac=1, split="train", transform=None, img_type="Frontal"):

        if data_dir is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )
        CHEXPERT_DATA_DIR = data_dir
        CHEXPERT_ORIGINAL_TRAIN_CSV = CHEXPERT_DATA_DIR + "/train.csv"
        CHEXPERT_TRAIN_CSV = CHEXPERT_DATA_DIR + "/train_split.csv"  # train split from train.csv
        CHEXPERT_VALID_CSV = CHEXPERT_DATA_DIR + "/valid_split.csv"  # valid split from train.csv
        CHEXPERT_TEST_CSV = CHEXPERT_DATA_DIR + "/valid.csv" # using validation set as test set (test set label hidden)
        CHEXPERT_TRAIN_DIR = CHEXPERT_DATA_DIR + "/train"
        CHEXPERT_TEST_DIR = CHEXPERT_DATA_DIR + "/valid"
        CHEXPERT_5x200 = CHEXPERT_DATA_DIR + "/chexpert_8x200.csv"
        CHEXPERT_VIEW_COL = "Frontal/Lateral"
        CHEXPERT_PATH_COL = "Path"
        CHEXPERT_SPLIT_COL = "Split"
        CHEXPERT_REPORT_COL = "Report Impression"

        CHEXPERT_TASKS = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Lesion",
            "Lung Opacity",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]
        CHEXPERT_COMPETITION_TASKS = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ]
        self.CHEXPERT_COMPETITION_TASKS = CHEXPERT_COMPETITION_TASKS

        # baseed on original chexpert paper
        CHEXPERT_UNCERTAIN_MAPPINGS = {
            "Atelectasis": 1,
            "Cardiomegaly": 0,
            "Consolidation": 0,
            "Edema": 1,
            "Pleural Effusion": 1,
        }

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(CHEXPERT_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(CHEXPERT_VALID_CSV)
        else:
            self.df = pd.read_csv(CHEXPERT_TEST_CSV)

        # sample data
        if sample_frac != 1 and split == "train":
            self.df = self.df.sample(frac=sample_frac)

        # filter image type
        if img_type != "All":
            self.df = self.df[self.df[CHEXPERT_VIEW_COL] == img_type]

        # get path
        self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
            lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        )

        # fill na with 0s
        self.df = self.df.fillna(0)

        # replace uncertains
        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
        super(CheXpertImageDataset, self).__init__(split, transform)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # get image
        img_path = row["Path"]
        x = self.read_from_jpg(img_path)

        # get labels
        y = list(row[self.CHEXPERT_COMPETITION_TASKS])
        y = torch.tensor(y)

        return {"image": x, "label": y}

    def __len__(self):
        return len(self.df)