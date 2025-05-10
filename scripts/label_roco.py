import os
import json
import torch
from tqdm import tqdm
from PIL import Image
import pandas as pd
from open_clip import create_model_from_pretrained, get_tokenizer

# Load model and preprocessing
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

image_dir = '/nethome/schopra47/nvme/bio/MedMoE/datasets/ROCOV2/'
data_dir = '/nethome/schopra47/nvme/bio/MedMoE/datasets/ROCOV2/'
output_file = os.path.join(data_dir, 'roco_data.csv')

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
train_df = pd.read_csv(os.path.join(data_dir, 'train_captions.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_captions.csv'))

images = train_df['ID'].tolist()
images.extend(test_df['ID'].tolist())

captions = train_df['Caption'].tolist()
captions.extend(test_df['Caption'].tolist())

batch_size = 64
save_every = 100
all_labels = []
for i in tqdm(range(0, len(images), batch_size)):
    batch_imgs = images[i:i + batch_size]
    batch_captions = captions[i:i + batch_size]
    batch_images = []

    for img, caption in zip(batch_imgs, batch_captions):
        
        try:
            img_path = os.path.join(image_dir, 'train', img+'.jpg')
            image = Image.open(img_path).convert('RGB')
            batch_images.append(preprocess(image))
        except Exception as e:
            img_path = os.path.join(image_dir, 'test', img+'.jpg')
            image = Image.open(img_path).convert('RGB')
            batch_images.append(preprocess(image))

    images_tensor = torch.stack(batch_images).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(images_tensor, text_tokens)
        logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
        predictions = torch.argmax(logits, dim=1)

    all_labels.extend(predictions.tolist())


roco_df = pd.DataFrame({'ID': images, 'caption': captions, 'label': all_labels})
roco_df.to_csv(output_file, index=False)