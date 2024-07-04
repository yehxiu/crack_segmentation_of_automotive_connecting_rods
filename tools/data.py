import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import glob
import cv2
import os


class ImageProcessor:
    def __init__(self, input_folder, output_folder, batch_size,midian:bool,erode:bool,dilate:bool):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.midian = midian
        self.erode = erode
        self.dilate = dilate
        # 確保輸出資料夾存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def batch_process_images(self):
        # 獲取所有影像檔案的列表
        files = [file for file in os.listdir(self.input_folder) if file.endswith('.jpg')]
        
        # 初始化批次計數器
        batch_count = 0
        
        # 分批處理影像
        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i+self.batch_size]
            for file in batch_files:
                image_path = os.path.join(self.input_folder, file)
                output_path = os.path.join(self.output_folder, file)
                
                # 讀取影像
                image = cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
                
                
                
                if self.midian == True:
                    image = cv2.medianBlur(image, 3)
                
                if self.erode == True:
                    kernel = np.ones((3, 3), np.uint8)
                    image = cv2.erode(image, kernel, iterations=1)

                if self.dilate == True:
                    kernel = np.ones((3, 3), np.uint8)
                    image = cv2.dilate(image, kernel, iterations=1)
                
                # 儲存處理後的影像
                cv2.imwrite(output_path, image)
            
            batch_count += 1
            print(f"Batch {batch_count} processed.")
        
        print("All batches processed.")

        

class tfData_Processor:
    def __init__(self, img_shape=(512, 512), batch_size=2):
        self.img_shape = img_shape
        self.batch_size = batch_size

    def decode_img(self, img_path, color_mode='rgb'):
        img = tf.io.read_file(img_path)
        if color_mode == 'rgb':
            img = tf.image.decode_jpeg(img, channels=3)
        else:
            img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, self.img_shape)
        return img / 255.0

    def decode_mask(self, mask_path):
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, self.img_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.cast(mask, tf.float32)
        mask = mask / 255.0
        mask = tf.cast(mask > 0.5, tf.float32)
        return mask

    def process_path(self, image_path, mask_path):
        image = self.decode_img(image_path)
        mask = self.decode_mask(mask_path)
        return image, mask

    def create_dataset(self, image_dir, mask_dir):
        image_paths = tf.io.gfile.glob(image_dir)
        mask_paths = tf.io.gfile.glob(mask_dir)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(self.process_path)
        dataset = dataset.batch(self.batch_size)
        return dataset

class DataGen_Processor:
    def __init__(self, img_shape=(512, 512), batch_size=2):
        self.img_shape = img_shape
        self.batch_size = batch_size

    def create_datagen(self):
        return ImageDataGenerator(rescale=1./255)

    def generate_data(self, image_path, mask_path, target_size=(512, 512)):
        image_datagen = self.create_datagen()
        mask_datagen = self.create_datagen()

        image_generator = image_datagen.flow_from_directory(
            image_path,
            target_size=target_size,
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode=None,
            seed = 42
        )

        mask_generator = mask_datagen.flow_from_directory(
            mask_path,
            target_size=target_size,
            color_mode='grayscale',
            batch_size=self.batch_size,
            class_mode=None,
            seed = 42
        )

        return zip(image_generator, mask_generator)

if __name__ == '__main__':
    print("The ImageProcessor class can perform preprocessing on images./n")
    print("The tfData_Processor class can help you prepare images./n")
    print("The DataGen_Processor class can perform data augmentation./n")