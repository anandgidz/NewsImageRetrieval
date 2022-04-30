from vocab import Vocabulary
import evaluation
import nltk

nltk.download('punkt')

evaluation.evalrank(model_path='./models/f30k_vse++_resnet_finetune.tar', headlines=False)