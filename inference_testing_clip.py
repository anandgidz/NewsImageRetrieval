import torch
import clip
from PIL import Image
from dataset import GoodNews
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

trainset = GoodNews(split = 'test', transform=preprocess)
image_index = 0
img, headline, caption = trainset[image_index]
image = img.unsqueeze(0).to(device)
text = clip.tokenize([headline]).to(device)

logits_per_image, logits_per_text = model(image, text)
print(logits_per_image)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T)
print(similarity)
