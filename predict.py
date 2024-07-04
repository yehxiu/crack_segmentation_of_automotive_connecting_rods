from tools import data,model,result
from keras.models import load_model
import numpy as np
import cv2


import glob

img_path = "/*.png"
imgs = glob.glob(img_path)

model = load_model("dg_0623_2.keras")
threshold = 0.9
for img in imgs:
    prossesed_img = result.image_process(img_name = img)
    #print(prossesed_img.shape)
    prediction = model.predict(prossesed_img)
    output_array = prediction.squeeze(axis=0).squeeze(axis=-1)
    binary_output = np.where(output_array > threshold, 1, 0)

    #mixed = result.mix_img(img,binary_output)
    #mixed.save("/home/yehxiu/crack/project_file/lun/00304_0.png")

    #contour , crack_number = result.crack_contour(img,binary_output)
    #cv2.imwrite("/home/yehxiu/crack/project_file/lun/00304_0.png",contour)

    #contour , box_number = result.crack_bounding_contour(img,binary_output)
    #cv2.imwrite("/home/yehxiu/crack/project_file/lun/00304_0.png",contour)




