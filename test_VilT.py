from transformers import ViltProcessor, ViltForImageAndTextRetrieval, ViltFeatureExtractor, BertTokenizerFast
import requests
from PIL import Image
from dataset import GoodNews
from tqdm import tqdm
import torch
import numpy

device = "cuda" if torch.cuda.is_available() else "cpu"

trainset = GoodNews(split = 'test')

image, headline, caption = trainset[0]
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco").to(device)

Tokenizer = BertTokenizerFast.from_pretrained("dandelin/vilt-b32-finetuned-coco")
ImageEncoder = ViltFeatureExtractor.from_pretrained("dandelin/vilt-b32-finetuned-coco")

# forward pass
scores = dict()
# for text in texts:
for i in tqdm(range(len(trainset))):
    image, text, _ = trainset[i]
    # prepare inputs
    # img_encoding = ViltFeatureExtractor(image)
    # img_encoding = ImageEncoder([image])
    txt_encoding = Tokenizer([text], truncation=True)
    # encoding = processor([image, image], texts, return_tensors="pt")
    
    # print(encoding)
    # outputs1 = model(**encoding)
    # print(torch.Tensor(numpy.array(img_encoding['pixel_values'])).to(device))
    # print(torch.tensor(numpy.array(img_encoding['pixel_mask']), dtype=torch.int32).to(device))

    # outputs = model(input_ids=torch.tensor(txt_encoding['input_ids']).to(device), 
    # token_type_ids=torch.tensor(txt_encoding['token_type_ids']).to(device), 
    # attention_mask = torch.tensor(txt_encoding['attention_mask']).to(device),
    # pixel_values = torch.tensor(numpy.array(img_encoding['pixel_values'])).to(device),
    # pixel_mask = torch.tensor(numpy.array(img_encoding['pixel_mask']), dtype=torch.int32).to(device))

    # print(outputs)
    # print(outputs1)

    # scores[text] = outputs.logits[0, :].item()

print(scores)