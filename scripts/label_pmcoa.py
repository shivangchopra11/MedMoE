# import os
# import json
# import torch
# from tqdm import tqdm
# from urllib.request import urlopen
# from PIL import Image
# from open_clip import create_model_from_pretrained, get_tokenizer

# # Load the model and config files from the Hugging Face Hub
# model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
# tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# image_dir = '/nethome/schopra47/nvme/bio/VLM/datasets/pmc_oa/caption_T060_filtered_top4_sep_v0_subfigures'
# data_dir = '/nethome/schopra47/nvme/bio/VLM/datasets/pmc_oa'


# # Zero-shot image classification
# template = 'this is a photo of '
# labels = [
#     'Ultrasound',
#     'Fluorescence',
#     'CT',
#     'MRI',
#     'X-ray',
#     'PET',
#     'DOT',
#     'Mitotic',
#     'Endoscope',
#     'Radioisotope',
#     'ENG'
# ]

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)
# model.eval()

# context_length = 256

# # images = torch.stack([preprocess(Image.open(os.path.join(image_dir, img))) for img in ]).to(device)
# texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)

# pmcoa_data = []

# with open(os.path.join(data_dir, 'filtered_data.jsonl'), 'r') as f:
#     for line in tqdm(f):
#         ann = json.loads(line.strip())
#         # print(ann)
#         # exit()
#         img_path = os.path.join(image_dir, ann['image'])
#         images = torch.stack([preprocess(Image.open(img_path))]).to(device)

#         with torch.no_grad():
#             image_features, text_features, logit_scale = model(images, texts)

#             logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)

#             label = torch.argmax(logits)
#             modality = labels[label.item()]

#             ann['pmcoa_label'] = label.item()
#             ann['modality'] = modality
#             pmcoa_data.append(ann)

#         if len(pmcoa_data) % 10000 == 0:
#             with open(os.path.join(data_dir, 'pmcoa_data_label.jsonl'), 'w') as f:
#                 for ann in pmcoa_data:
#                     f.write(json.dumps(ann) + '\n')



# # with open(os.path.join(data_dir, 'pmcoa_data_label.jsonl'), 'w') as f:
# #     for ann in pmcoa_data:
# #         f.write(json.dumps(ann) + '\n')


import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# Load model and preprocessing
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

image_dir = '/nethome/schopra47/nvme/bio/VLM/datasets/pmc_oa/caption_T060_filtered_top4_sep_v0_subfigures'
data_dir = '/nethome/schopra47/nvme/bio/VLM/datasets/pmc_oa'
output_file = os.path.join(data_dir, 'pmcoa_data_label_four.jsonl')

# Classification labels and template
labels = [
    'X-ray',
    'CT',
    'MRI',
    'Ultrasound',
    'Pathology',
    'Fundus'
]
template = 'this is a photo of '
context_length = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
model.eval()

# Encode all label prompts once
text_tokens = tokenizer([template + l for l in labels], context_length=context_length).to(device)

# Load and group annotations
with open(os.path.join(data_dir, 'train.jsonl'), 'r') as f:
    annotations = [json.loads(line.strip()) for line in f]

with open(os.path.join(data_dir, 'test.jsonl'), 'r') as f:
    test_annotations = [json.loads(line.strip()) for line in f]

annotations.extend(test_annotations)

batch_size = 64
save_every = 100
pmcoa_data = []

for i in tqdm(range(0, len(annotations), batch_size)):
    batch_anns = annotations[i:i + batch_size]
    batch_images = []

    for ann in batch_anns:
        img_path = os.path.join(image_dir, ann['image'])
        try:
            image = Image.open(img_path).convert('RGB')
            batch_images.append(preprocess(image))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            batch_images.append(torch.zeros_like(preprocess(Image.new('RGB', (224, 224)))))

    images_tensor = torch.stack(batch_images).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(images_tensor, text_tokens)
        logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
        predictions = torch.argmax(logits, dim=1)

    for ann, pred_idx in zip(batch_anns, predictions):
        label_id = pred_idx.item()
        ann['pmcoa_label'] = label_id
        ann['modality'] = labels[label_id]
        pmcoa_data.append(ann)

    # Save intermediate results
    if len(pmcoa_data) % save_every == 0 or i + batch_size >= len(annotations):
        with open(output_file, 'w') as out_f:
            for record in pmcoa_data:
                out_f.write(json.dumps(record) + '\n')
        