import json
import csv, io, os, uuid, numpy as np
from PIL import Image
import pandas as pd
import webdataset as wds
from tqdm import tqdm

base_data_dir = '/nethome/schopra47/nvme/bio/MedMoE/datasets/'
output_file = '/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/pmcoa_webdataset/dataset-%06d.tar'
max_samples_per_shard = 10000


annotations = os.path.join(base_data_dir, 'pmc_oa', 'pmcoa_data_label_four.jsonl')

writer = wds.ShardWriter(output_file, maxcount=max_samples_per_shard)

with open(annotations, 'r') as f:
    data = [json.loads(line.strip()) for line in f]
    
for idx, row in tqdm(enumerate(data), total=len(data)):
    img_id = row['image']
    caption = row['caption']

    label = row['pmcoa_label']
    
    img_path = os.path.join(base_data_dir, 'pmc_oa', 'caption_T060_filtered_top4_sep_v0_subfigures', img_id)

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

writer.close()
