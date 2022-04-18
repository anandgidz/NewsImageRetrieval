import torch
import clip
from PIL import Image
from dataset import GoodNews
import numpy as np
from tqdm import tqdm
import pickle

with open('test_set_image_features.pickle', 'rb') as handle:
    batchwise_image_features = pickle.load(handle)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

testset = GoodNews(split = 'test', transform=preprocess)
top10_accuracy = 0
top5_accuracy = 0
top1_accuracy = 0

progression_bar = tqdm(range(len(testset)))

for i in (progression_bar):
    progression_bar.set_description(f'T10: {round(top10_accuracy/(i+1), 3)}, T5: {round(top5_accuracy/(i+1), 3)}, T1: {round(top1_accuracy/(i+1), 3)}')
    _, headline, caption = testset[i]
    text = clip.tokenize([caption]).to(device)
    text_features = model.encode_text(text)

    text_features /= text_features.norm(dim=-1, keepdim=True).detach()

    similarities = []
    for image_features_batch in batchwise_image_features:

        image_features_batch = image_features_batch.to(device)
        image_features_batch /= image_features_batch.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features_batch @ text_features.T)
        similarity = similarity.cpu()

        for x in similarity:
            similarities.append(x[0].item())

    similarities = np.array(similarities)
    sorted_indexes = np.argsort(similarities)
    if sorted_indexes[-1] == i:
        top1_accuracy += 1
    if i in sorted_indexes[-5:]:
        top5_accuracy += 1
    if i in sorted_indexes[-10:]:
        top10_accuracy += 1


print(top10_accuracy)
print(top5_accuracy)
print(top1_accuracy)

print(top10_accuracy/len(testset))
print(top5_accuracy/len(testset))
print(top1_accuracy/len(testset))
