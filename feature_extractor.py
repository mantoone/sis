from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import numpy as np
import tensorflow as tf

# Do not use all memory!
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


class FeatureExtractor:
    def __init__(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        base_model.summary()
        self.model = base_model
        self.graph = tf.get_default_graph()

    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        with self.graph.as_default():
            feature = self.model.predict(x).flatten()  # (1, 4096) -> (4096, )
            return feature / np.linalg.norm(feature)  # Normalize
