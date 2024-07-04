from tools import data,model,result

from keras.callbacks import LearningRateScheduler,ModelCheckpoint
from keras.metrics import IoU

data_processor = data.DataGen_Processor()

train_img_path = '/data_gen_img'
train_mask_path = '/data_gen_mask'

dataset = data_processor.generate_data(train_img_path,train_mask_path,batch_size = 2)

my_model = model.Unet(num_classes = 1,IMG_HEIGHT=512,IMG_WIDTH=512,IMG_CHANNELS=3)
my_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[IoU(num_classes=2, target_class_ids=[1])])
my_model.summary()

lr_callback =  [LearningRateScheduler(lambda epoch:1e-4*10**(1/(epoch+1)))]

checkpoint = ModelCheckpoint(
    filepath = 'best_model.keras',  # 保存的模型文件名稱
    monitor='io_u',  # 監控的指標
    save_best_only=True,  # 只保存性能最好的模型
    mode='max',  # 尋找最小的val_loss
    verbose=1  # 顯示保存信息
)

history = my_model.fit(dataset , epochs=1,callbacks=[lr_callback])

#model.save("")

draw_matlab = result.TrainingVisualizer(history= history,save_dir="")

draw_matlab.draw_iou(show_val_iou=False)
draw_matlab.draw_loss(show_val_loss=False)
draw_matlab.draw_lr
