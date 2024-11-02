import os
import cv2
import numpy as np
from .preprocess import create_operators
from .utils.get_image_list import get_image_list
from .utility import create_predictor, get_config
from .utils import logger
import json
import sklearn.preprocessing

class Embedding:

    __instance = None

    @staticmethod
    def getInstance():
        if Embedding.__instance == None:
            Embedding()
        return Embedding.__instance
    
    def __init__(self):
      if Embedding.__instance != None:
         """ Raise exception if init is called more than once. """
         raise Exception("This class is a singleton!")
      else:
         self.load_model()
         Embedding.__instance = self

    def load_model(self):
        self.config = get_config('/media/dangph/data-1tb1/data_donghonuoc/clustering/test_sku_embedding/model_lib/utils/inference_general.yaml')
        self.preprocess_ops = create_operators(self.config["RecPreProcess"][
            "transform_ops"])
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            create_predictor(self.config, 'embed')
    
        
    def postprocess(self, batch_output, batch_names):
        results =[]
        for number, result_dict in enumerate(batch_output):
            result = {'image_name': batch_names[number], 'vector' : result_dict.tolist()}
            results.append(result)
        return results


    def __call__(self, images, batch_names, feature_normalize=True):

        input_names = self.predictor.get_inputs()[0].name
        output_names = self.predictor.get_outputs()[0].name
        if not isinstance(images, (list, )):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)

        batch_output = self.predictor.run(
            output_names=[output_names],
            input_feed={input_names: image})[0]

        if feature_normalize:
            # feas_norm = np.sqrt(
            #     np.sum(np.square(batch_output), axis=1, keepdims=True))
            # batch_output = np.divide(batch_output, feas_norm)
            batch_output = [sklearn.preprocessing.normalize([vector])[0] for vector in batch_output]
        results = self.postprocess(batch_output, batch_names)
        return results


def main(config):
    embedding = Embedding()
    image_list = get_image_list("")

    batch_imgs = []
    batch_names = []
    cnt = 0

    for idx, img_path in enumerate(image_list):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(
                "Image file failed to read and has been skipped. The path: {}".
                format(img_path))
        else:
            img = img[:, :, ::-1]
            batch_imgs.append(img)
            img_name = os.path.basename(img_path)
            batch_names.append(img_name)
            cnt += 1

        if cnt % config["Global"]["batch_size"] == 0 or (idx + 1
                                                         ) == len(image_list):
            if len(batch_imgs) == 0:
                continue
            
            batch_results = embedding(batch_imgs,batch_names)
            batch_imgs = []
            batch_names = []
            print(batch_results )
