import os, json, pickle, random
import torch.utils.data
from PIL import Image


class GoodNews(torch.utils.data.Dataset):

    def __init__(self, split='train', transform=None):
        self.transform = transform
        f = open(f"./{split}_set.json")
        data = json.load(f)
        self.img_headline_pair = [(f'./resized/{k}', data[k]['headline'], data[k]['caption']) for k in data.keys()]
        self.img_headline_pair = sorted(self.img_headline_pair, key=lambda x: x[0])

        f.close()

    def __getitem__(self, index):
        image = Image.open(self.img_headline_pair[index][0]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.img_headline_pair[index][1], self.img_headline_pair[index][2]

    # Return the total number of samples.
    def __len__(self):
        return len(self.img_headline_pair)


if __name__ == "__main__":

    trainset = GoodNews(split = 'train')
    image_index = 1  # Feel free to change this.

    print('This dataset has {0} training images'.format(len(trainset)))

    # 2. Datasets need to implement the  __getitem__ method for this to work.
    img, headline, caption = trainset[image_index]  # Returns image and label.

    print('Image {0} has headline {1} and caption {2}'.format(image_index, headline, caption))
    img.show()
