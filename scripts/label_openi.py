import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# Load model and preprocessing
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

image_dir = '/nethome/schopra47/nvme/bio/MedMoE/datasets'
data_dir = '/nethome/schopra47/nvme/bio/MedMoE/datasets/openI'
output_file = os.path.join(data_dir, 'openi_labeled.jsonl')

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
with open(os.path.join(data_dir, 'openI.jsonl'), 'r') as f:
    annotations = [json.loads(line.strip()) for line in f]

batch_size = 64
save_every = 100
openi_data = []

for i in tqdm(range(0, len(annotations), batch_size)):
    print(annotations[i])
    batch_anns = annotations[i:i + batch_size]
    batch_images = []

    for ann in batch_anns:
        img_path = image_dir + ann['image']
        print(image_dir)
        print(ann['image'])
        print(img_path)

        image = Image.open(img_path).convert('RGB')
        batch_images.append(preprocess(image))

    images_tensor = torch.stack(batch_images).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(images_tensor, text_tokens)
        logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
        predictions = torch.argmax(logits, dim=1)

    for ann, pred_idx in zip(batch_anns, predictions):
        label_id = pred_idx.item()
        ann['label'] = label_id
        ann['modality'] = labels[label_id]
        openi_data.append(ann)

    # Save intermediate results
    if len(openi_data) % save_every == 0 or i + batch_size >= len(annotations):
        with open(output_file, 'w') as out_f:
            for record in openi_data:
                out_f.write(json.dumps(record) + '\n')
        