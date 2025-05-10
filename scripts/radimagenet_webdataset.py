import json
import csv, io, os, uuid, numpy as np
from PIL import Image
import pandas as pd
import webdataset as wds
from tqdm import tqdm

base_data_dir = '/nethome/schopra47/nvme/bio/MedMoE/datasets/'
output_file = '/nethome/schopra47/nvme/bio/MedMoE/datasets/unimed/radimagenet_webdataset/dataset-%06d.tar'
max_samples_per_shard = 10000

annotations = pd.read_csv(os.path.join(base_data_dir, 'radiology_ai', 'radimagenet_with_captions_training_set.csv'))

writer = wds.ShardWriter(output_file, maxcount=max_samples_per_shard)

for idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
        img_id = json.loads(row['filename'].replace("'", '"'))[0]
        all_captions = json.loads(row['captions'].replace("'", '"'))
        caption = all_captions[0]

        if 'CT' in img_id:
            label = 1
        elif 'MR' in img_id:
            label = 2
        elif 'UT' in img_id:
            label = 3
        
        img_path = os.path.join(base_data_dir, img_id)

        img = Image.open(img_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="jpeg", quality=95)
        img_bytes = buf.getvalue() 

        for l_idx, l_caption in enumerate(all_captions):
            sample = {
                "__key__": f"{idx:08d}_{l_idx:02d}",              # reproducible key
                "jpg": img_bytes,                     # image payload
                "txt": l_caption.strip(),        # caption
                "cls": str(label)  # 8‑byte label
            }
            writer.write(sample)

        # sample = {
        #     "__key__": f"{idx:08d}",              # reproducible key
        #     "jpg": img_bytes,                     # image payload
        #     "txt": caption.strip(),        # caption
        #     "cls": str(label)  # 8‑byte label
        # }
        # writer.write(sample)

writer.close()
