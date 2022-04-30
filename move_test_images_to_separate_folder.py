import json
import shutil
from tqdm import tqdm

f = open("./test_set.json")
data = json.load(f)

for k in tqdm(data.keys()):
    shutil.copy(f'./resized/{k}', f'./test_images/{k}')

f.close()