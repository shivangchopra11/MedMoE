import json
import csv, io, os, uuid, numpy as np
from PIL import Image
import pandas as pd
import webdataset as wds
from tqdm import tqdm

base_data_dir = '/nethome/schopra47/nvme/bio/MedMoE/datasets/'
output_file = '/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/quilt_webdataset/dataset-%06d.tar'
max_samples_per_shard = 10000

annotations = pd.read_csv(os.path.join(base_data_dir, 'quilt1m', 'quilt_labeled.csv'))

writer = wds.ShardWriter(output_file, maxcount=max_samples_per_shard)

for idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
    try:
        img_id = row['image_path']
        caption = row['caption']
        label = row['label']
        
        img_path = os.path.join(base_data_dir, 'quilt1m', 'quilt_1m', img_id)

        img = Image.open(img_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="jpeg", quality=95)
        img_bytes = buf.getvalue() 

        sample = {
            "__key__": f"{idx:08d}",              # reproducible key
            "jpg": img_bytes,                     # image payload
            "txt": caption.strip(),        # caption
            "cls": str(label)  # 8‑byte label
        }
        writer.write(sample)
    except Exception as e:
        continue

writer.close()
