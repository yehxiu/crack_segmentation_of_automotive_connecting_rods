This project is a crack segmentation task based on AttU-Net.
We built a new dataset of automotive connecting rods.
Here is the introduction of the files in the "tools" directory:

1. data.py: This file contains a simple image preprocessing class called "ImageProcessor," which you can use to apply median blur, erosion, or dilation to images. It also includes functions for data preprocessing. The "tfData_Processor" class is used to prepare the dataset for training. If you need to perform data augmentation, you can use the "DataGen_Processor" class.

2. model.py: This file contains the model used in this project. You can call the "Unet" function to utilize it.

3. result.py: After training your model, if you want to see its performance, you can use the "TrainingVisualizer" class to display the "loss," "iou," and "lr." For predictions, you can use "crack_bounding_contour," "crack_contour," or "mix_img" to show the cracks in different ways.

in the main page,"train_datagen.py" and "train_tfdata.py" are both the codes used to train the model. 
