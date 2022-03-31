# This is an in-progress repository, [incomplete]

# cv-age

### Age and Sex Prediction using Deep Learning

#### Model Structures Used
- Convolutional Neural Network
  - The Sex Classification Model uses a similar structure to the one presented in the paper below
  - [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- Residual Neural Network
  - There are several implementations of RNNs in this repository
  - RNN reference: [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
  - There are personally made implementations of ResNet34 and ResNet50 as outlined in the paper
  - Additionally, testing was done using a transfer learning structure with the tensorflow pretrained ResNet50 Model [Tensorflow ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50)
- Transformers Model
  - Testing was done using Vision Transformer (ViT) model outlined originally in the paper below
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)
  - pretrained ViT Base Model used for testing [Vision Transformer (base-sized model)](https://huggingface.co/google/vit-base-patch16-224)

#### Datasets
- UTKFace
  - 20k+ images [UTKFace Source](https://susanqq.github.io/UTKFace/)
- IMDB
  - 460k+ images [IMDB Source](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- Wikipedia
  - 62k+ images [Wikipedia Source](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

#### Dependencies
- torch
  - python -m pip install torch
- transformers
  - python -m pip install transformers
- OpenCV
  - python -m pip install opencv-python
- tensorflow
  - python -m pip install tensorflow
- numpy
  - python -m pip install numpy
- pillow
  - python3 -m pip install --upgrade Pillow
