import os
import json
import random
from tqdm import tqdm
import math

f = open("./captioning_dataset.json")
data = json.load(f)

print(len(data))
# complete_dataset = {}
img_headlines = []
n = 0
for k in tqdm(data):
    print(data[k])
    break
    # if 'print_headline' not in data[k]['headline']:
    #     n += 1
    #     continue
    # print(data[k]['headline'])
    # img_headlines.append((f'{k}_{0}.jpg', data[k]['headline']['main']))
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
    if 'main' not in data[k]['headline']:
        continue
    if not os.path.exists(f'./resized/{k}_{0}.jpg'):
        n += 1

print(n)