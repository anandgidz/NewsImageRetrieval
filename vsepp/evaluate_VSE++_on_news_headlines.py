from vocab import Vocabulary
import evaluation
import nltk
nltk.download('punkt')

evaluation.evalrank(model_path='./vggfull/model_best.pth.tar', headlines=True)