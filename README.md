The Dataset can be found from [GoodNews](https://github.com/furkanbiten/GoodNews), which contains NYTimes articles with headlines, images and their captions.

test_set.json contains image and headline, caption pairs.

save_test_set_image_features_clip.py pickles the encodings of all the images in the test set encoded using CLIPs image encoder. (ViT-B/32 in this case)

test_accuracy_clip.py tests r@1, r@5, r@10 of CLIP on our test set.

VSE++ pretrained models and vocab can be found at [vsepp](https://github.com/fartashf/vsepp). The runs.tar file contains the pretrained models.

vsepp/evaluate_VSE++_on_news_captions.py evaluates VSE++ on our test sets captions
vsepp/evaluate_VSE++_on_news_headlines.py evaluates VSE++ on our test sets headlines
Path errors may need to be resolved for the above two scripts
