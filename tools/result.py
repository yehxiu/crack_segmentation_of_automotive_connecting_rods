import cv2 as cv
from PIL import Image, ImageDraw
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

def image_process(img_name,gray = False,img_size = (512,512)):
   img = cv.imread(img_name)
   if gray == True:
      img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
   result_img = cv.resize(img,img_size)
   result_img = np.expand_dims(result_img, axis=0)
   return result_img
   

def getWandH(img_name):
  img = cv.imread(img_name)
  return img.shape[:2]

def mix_img(img_name,mask):
  restore_size = getWandH(img_name)
  w = restore_size[1]
  h = restore_size[0]
  original_image = Image.open(img_name).convert('RGB')
  original_array = np.array(original_image)
  binary_image = Image.fromarray((mask * 255).astype(np.uint8))
  resize_mask = binary_image.resize((w,h),Image.Resampling.LANCZOS)

  binary_output_resized = np.array(resize_mask) // 255
  
  original_array[binary_output_resized == 1] = [255, 0, 0]
  modified_image = Image.fromarray(original_array)
  return modified_image

def crack_contour(img_name,mask):
  restore_size = getWandH(img_name)
  w = restore_size[1]
  h = restore_size[0]
  original_image = Image.open(img_name).convert('RGB')
  original_array = np.array(original_image)
  binary_image = Image.fromarray((mask * 255).astype(np.uint8))
  resize_mask = binary_image.resize((w,h),Image.Resampling.LANCZOS) 
  resize_mask_array = np.array(resize_mask)
  contours, _ = cv.findContours(resize_mask_array.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  filtered_contours = [contour for contour in contours if cv.contourArea(contour) > 10]

  cv.drawContours(original_array, contours, -1, (0, 255, 0), 2)

  return original_array,len(filtered_contours)

def crack_bounding_contour(img_name,mask):
  restore_size = getWandH(img_name)
  w = restore_size[1]
  h = restore_size[0]
  original_image = Image.open(img_name).convert('RGB')
  original_array = np.array(original_image)
  binary_image = Image.fromarray((mask * 255).astype(np.uint8))
  resize_mask = binary_image.resize((w,h),Image.Resampling.LANCZOS) 
  resize_mask_array = np.array(resize_mask)
  contours, _ = cv.findContours(resize_mask_array.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  filtered_contours = [contour for contour in contours if cv.contourArea(contour) > 10]

  for contour in filtered_contours:
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(original_array, (x, y), (x + w, y + h), (0, 255, 0), 4)  

  return original_array,len(filtered_contours)

class TrainingVisualizer:
    def __init__(self, history,save_dir):
        self.history = history.history
        self.epochs = range(1, len(self.history['loss']) + 1)
        self.save_dir = save_dir
    
    def draw_loss(self, show_val_loss=True):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.history['loss'], 'bo-', label='Training loss')
        if show_val_loss and 'val_loss' in self.history:
            plt.plot(self.epochs, self.history['val_loss'], 'r*-', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.save_dir}/loss.png')
        plt.close()

    def draw_iou(self, show_val_iou=True):
        plt.figure(figsize=(10, 6))
        if 'io_u' in self.history:
            plt.plot(self.epochs, self.history['io_u'], 'bo-', label='Training IoU')
        if show_val_iou and 'val_io_u' in self.history:
            plt.plot(self.epochs, self.history['val_io_u'], 'r*-', label='Validation IoU')
        plt.title('Training and Validation IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.save_dir}/iou.png')
        plt.close()

    def draw_lr(self):
        plt.figure(figsize=(10, 6))
        if 'lr' in self.history:
            plt.plot(self.epochs, self.history['lr'], 'bo-', label='Training Learning Rate')
        plt.title('Training Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.savefig(f'{self.save_dir}/lr.png')
        plt.close()

if __name__ == '__main__':
  #visualizer = TrainingVisualizer(history,save_dir)
  #visualizer.draw_loss(show_val_loss=True)  
  #visualizer.draw_iou(show_val_iou=True)    
  #visualizer.draw_lr()     
  pass


