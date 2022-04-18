import os
import json
import random
from tqdm import tqdm
import math

f = open("./captioning_dataset.json")
data = json.load(f)

print(len(data))
img_headlines = []
for k in tqdm(data):
    if 'main' not in data[k]['headline'] or not os.path.exists(f'./resized/{k}_{0}.jpg'):
        continue
    img_headlines.append((f'{k}_{0}.jpg', data[k]['headline']['main'], data[k]['images']['0']))

f.close()

print(len(img_headlines))
print(img_headlines[:5])
random.shuffle(img_headlines)
print(img_headlines[:5])

# train_set_size = math.ceil(len(img_headlines) * 0.8)
# val_set_size = math.ceil(len(img_headlines) * 0.1)

# train_set = img_headlines[: train_set_size]
# val_set = img_headlines[train_set_size: train_set_size + val_set_size]
# test_set = img_headlines[train_set_size + val_set_size:]


val_set = img_headlines[:5000]
test_set = img_headlines[5000:10000]
train_set = img_headlines[10000:]

print(len(val_set))
print(len(test_set))

train_set_d = {img: {'headline': headline, 'caption': caption} for img, headline, caption in train_set}
val_set_d = {img: {'headline': headline, 'caption': caption} for img, headline, caption in val_set}
test_set_d = {img: {'headline': headline, 'caption': caption} for img, headline, caption in test_set}

with open("train_set.json", "w") as outfile:
    json.dump(train_set_d, outfile)

with open("val_set.json", "w") as outfile:
    json.dump(val_set_d, outfile)

with open("test_set.json", "w") as outfile:
    json.dump(test_set_d, outfile)

