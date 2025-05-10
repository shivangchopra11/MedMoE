import csv, io, os, uuid, numpy as np
from PIL import Image
import pandas as pd
import webdataset as wds
from tqdm import tqdm

base_data_dir = '/nethome/schopra47/nvme/bio/MedMoE/datasets/ROCOV2'
output_file = '/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/roco_webdataset/dataset-%06d.tar'
max_samples_per_shard = 10000

annotations = pd.read_csv(os.path.join(base_data_dir, 'roco_data.csv'))

writer = wds.ShardWriter(output_file, maxcount=max_samples_per_shard)

for idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
        img_id = row['ID']
        if 'train' in img_id:
            img_path = os.path.join(base_data_dir, 'train', img_id + '.jpg')
        elif 'valid' in img_id:
            img_path = os.path.join(base_data_dir, 'valid', img_id + '.jpg')
        else:
            img_path = os.path.join(base_data_dir, 'test', img_id + '.jpg')

        img = Image.open(img_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="jpeg", quality=95)
        img_bytes = buf.getvalue() 

        caption = row['caption']
        label = row['label']

        # print(img_path, caption, label)

        sample = {
            "__key__": f"{idx:08d}",              # reproducible key
            "jpg": img_bytes,                     # image payload
            "txt": caption.strip(),        # caption
            "cls": str(label)  # 8‑byte label
        }
        writer.write(sample)

writer.close()
