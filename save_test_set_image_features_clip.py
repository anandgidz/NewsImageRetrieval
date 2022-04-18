import torch
import clip
from PIL import Image
from dataset import GoodNews
import numpy as np
from tqdm import tqdm
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

trainset = GoodNews(split = 'test', transform=preprocess)

loader = torch.utils.data.DataLoader(trainset, 
                                    batch_size = 30, 
                                    shuffle = False)

all_image_features = []
for imgs, _, _ in tqdm(loader):
    imgs = imgs.to(device)
    with torch.no_grad():
        image_features = model.encode_image(imgs)
        all_image_features.append(image_features)

with open('test_set_image_features.pickle', 'wb') as handle:
    pickle.dump(all_image_features, handle)
