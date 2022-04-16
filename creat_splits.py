import os
import json
import cv2
import random
from tqdm import tqdm
import math

f = open("./captioning_dataset.json")
data = json.load(f)

print(len(data))
# complete_dataset = {}
img_headlines = []
for k in tqdm(data):
    if 'main' not in data[k]['headline']:
        continue
    img_headlines.append((f'{k}_{0}.jpg', data[k]['headline']['main']))
    # i = 0
    # while True:
    #     if (not os.path.exists(f'./resized/{k}_{i}.jpg')):
    #         break
    #     img_headlines.append((f'{k}_{i}.jpg', data[k]['headline']['main']))
    #     # complete_dataset[f'{k}_{i}.jpg'] = data[k]['headline']['main']
    #     # img = cv2.imread(f'./resized/{k}_{i}.jpg')
    #     i += 1
        #
        # cv2.imshow('image', img)
        # print(data[k]['headline']['main'])
        # cv2.waitKey(0)

f.close()

print(len(img_headlines))
print(img_headlines[:5])
random.shuffle(img_headlines)
print(img_headlines[:5])

train_set_size = math.ceil(len(img_headlines) * 0.8)
val_set_size = math.ceil(len(img_headlines) * 0.1)

train_set = img_headlines[: train_set_size]
val_set = img_headlines[train_set_size: train_set_size + val_set_size]
test_set = img_headlines[train_set_size + val_set_size:]

print(len(val_set))
print(len(test_set))

train_set_d = {img: headline for img, headline in train_set}
val_set_d = {img: headline for img, headline in val_set}
test_set_d = {img: headline for img, headline in test_set}

with open("train_set.json", "w") as outfile:
    json.dump(train_set_d, outfile)

with open("val_set.json", "w") as outfile:
    json.dump(val_set_d, outfile)

with open("test_set.json", "w") as outfile:
    json.dump(test_set_d, outfile)

# print(complete_dataset)

