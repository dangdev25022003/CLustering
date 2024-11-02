import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))
import glob
import time
import cv2
from model_lib.utils import logger
from model_lib.utility import get_config
from model_lib.utils.get_image_list import get_image_list
from model_lib.embedding import Embedding


if __name__ == '__main__':

    config = get_config("/mnt/sdb1/data_donghonuoc/clustering/test_sku_embedding/model_lib/utils/inference_general.yaml")
    # Initialize model
    embedding = Embedding.getInstance()

    image_list = get_image_list('/mnt/sdb1/data_donghonuoc/t7')

    batch_imgs = []
    batch_names = []
    cnt = 0
    
    for idx, img_path in enumerate(image_list):
        img = cv2.imread(img_path)
        # print(img.shape)
        # f = open('img.txt', 'w')
        # f.write(str(img[1000].tolist()))
        # f.close()
        if img is None:
            logger.warning(
                "Image file failed to read and has been skipped. The path: {}".
                format(img_path))
        else:
            # img = img[:, :, ::-1]
            batch_imgs.append(img)
            img_name = os.path.basename(img_path)
            batch_names.append(img_name)
            cnt += 1

        if cnt % 10 == 0 or (idx + 1) == len(image_list):
            if len(batch_imgs) == 0:
                continue
            t1 = time.time()
            batch_results = embedding(batch_imgs, batch_names)
            print((time.time()-t1 )/10)
            batch_imgs = []
            # batch_names = []
            # print(batch_results)
    

