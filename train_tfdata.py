from tools import data,model,result

from keras.callbacks import LearningRateScheduler
from keras.metrics import IoU

data_processor = data.tfData_Processor()
image_path = "/*.png"
mask_path = "/*.png"

dataset = data_processor.create_dataset(image_dir = image_path, mask_dir = mask_path)

my_model = model.Unet(num_classes = 1,IMG_HEIGHT=512,IMG_WIDTH=512,IMG_CHANNELS=3)
my_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[IoU(num_classes=2, target_class_ids=[1])])
my_model.summary()

lr_callback =  [LearningRateScheduler(lambda epoch:1e-4*10**(1/(epoch+1)))]
history = my_model.fit(dataset , epochs=1,callbacks=[lr_callback])

#model.save("")

draw_matlab = result.TrainingVisualizer(history= history,save_dir="result_img")

draw_matlab.draw_iou(show_val_iou=False)
draw_matlab.draw_loss(show_val_loss=False)
draw_matlab.draw_lr
