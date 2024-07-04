from keras.layers import *
from keras.initializers import HeNormal,RandomNormal
from keras.models import *

from keras.metrics import IoU,MeanIoU

def VGG16_Block1(img_input,kernal_size):
  x = Conv2D(kernal_size,(3,3),
            activation = 'relu', #"sigmoid"
            padding = 'same',
            kernel_initializer = RandomNormal(stddev = 0.02),
            )(img_input)

  x = Conv2D(kernal_size,(3,3),
            activation = 'relu',
            padding = 'same',
            kernel_initializer = RandomNormal(stddev = 0.02),
            )(x)

  sc = x

  x = MaxPooling2D((2,2), strides=(2,2))(x)

  return sc,x


def VGG16_Block2(img_input,kernal_size,The5th_Block = False):
  x = Conv2D(kernal_size,(3,3),
            activation = 'relu',
            padding = 'same',
            kernel_initializer = RandomNormal(stddev = 0.02),
            )(img_input)

  x = Conv2D(kernal_size,(3,3),
            activation = 'relu',
            padding = 'same',
            kernel_initializer = RandomNormal(stddev = 0.02),
            )(x)

  x = Conv2D(kernal_size,(3,3),
            activation = 'relu',
            padding = 'same',
            kernel_initializer = RandomNormal(stddev = 0.02),
            )(x)


  if The5th_Block == False:
    sc = x
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    return sc,x
  else:
    return x

def VGG16(img_input):
  sc1,block1 = VGG16_Block1(img_input,64) #sc1 : 512,512,3 -> 512,512,64 ,block1 : 512,512,64 -> 256,256,64
  sc2,block2 = VGG16_Block1(block1,128) #sc2 : 256,256,64 -> 256,256,128 ,block2 : 256,256,128 -> 128,128,128
  sc3,block3 = VGG16_Block2(block2,256,The5th_Block = False) #sc3 : 128,128,128 -> 128,128,256 ,block3 : 128,128,256 -> 64,64,256
  sc4,block4 = VGG16_Block2(block3,512,The5th_Block = False) #sc4 : 64,64,256 -> 64,64,512 ,block4 : 64,64,512 -> 32,32,512
  block5 = VGG16_Block2(block4,512,The5th_Block = True) # block5 = 32,32,512 -> 32,32,512

  return sc1,sc2,sc3,sc4,block5

def attention_gate(input_tensor,gating_signal,inter_channel):
  theta_x = Conv2D(inter_channel,(2,2),padding = 'same')(input_tensor)
  phi_g = Conv2D(inter_channel, (1,1), padding='same')(gating_signal)

  concat = Add()([theta_x, phi_g])
  act = Activation('relu')(concat)
  psi = Conv2D(1, (1,1), padding='same')(act)
  sigmoid = Activation('sigmoid')(psi)
#  upsample_psi = UpSampling2D(size=(2, 2))(sigmoid)


  # 乘上原始輸入張量以聚焦到更重要的部分
  output = Multiply()([input_tensor, sigmoid])
  return output

def Unet(num_classes = 1,IMG_HEIGHT=256,IMG_WIDTH=256,IMG_CHANNELS=3): #黑白CHANNELS = 1 , RGB = 3
  inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  sc1,sc2,sc3,sc4,block5 = VGG16(inputs)

  channels = [64 ,128 ,256 ,512]



  block5_up = UpSampling2D(size=(2, 2))(block5) # 32, 32, 512 -> 64, 64, 512

  sc4_gating = attention_gate(sc4, block5_up, 512)

  block4_up = Concatenate(axis=3)([sc4_gating, block5_up]) # 64, 64, 512 + 64, 64, 512 -> 64, 64, 1024

  block4_up = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(block4_up)
  block4_up = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(block4_up) #64, 64, 1024 -> 64, 64, 512

  block4_up = UpSampling2D(size=(2, 2))(block4_up) # 64, 64, 512 -> 128, 128, 512


  sc3_gating = attention_gate(sc3, block4_up, 256)

  block3_up = Concatenate(axis=3)([sc3_gating, block4_up]) # 128, 128, 256 + 128, 128, 512 -> 128, 128, 768

  block3_up = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(block3_up)
  block3_up = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(block3_up) # 128, 128, 768 -> 128, 128, 256

  block3_up = UpSampling2D(size=(2, 2))(block3_up) # 128, 128, 256 -> 256, 256, 256


  sc2_gating = attention_gate(sc2, block3_up, 128)

  block2_up = Concatenate(axis=3)([sc2_gating, block3_up]) # 256, 256, 256 + 256, 256, 128 -> 256, 256, 384

  block2_up = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(block2_up)
  block2_up = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(block2_up) # 256, 256, 384 -> 256, 256, 128

  block2_up = UpSampling2D(size=(2, 2))(block2_up) # 256, 256, 128 -> 512, 512, 128


  sc1_gating = attention_gate(sc1, block2_up, 64)

  block1_up = Concatenate(axis=3)([sc1_gating, block2_up]) # 512, 512, 128 + 512, 512, 64 -> 512, 512, 192

  block1_up = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(block1_up)
  block1_up = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(block1_up) # 512, 512, 192 -> 512, 512, 64

  output = Conv2D(num_classes, 1,(1,1), activation="sigmoid")(block1_up)  # 512, 512, 64 -> 512, 512, num_classes #softmax or sigmoid


  model = Model(inputs=inputs, outputs=output)
  return model

if __name__ == '__main__':
    model = Unet(num_classes = 1,IMG_HEIGHT=512,IMG_WIDTH=512,IMG_CHANNELS=3) #黑白IMG_CHANNELS=1,彩色IMG_CHANNELS=3
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[IoU(num_classes=2, target_class_ids=[1])]) # loss = "Dice" "sparse_categorical_crossentropy" "categorical_crossentropy 要做one-hot encoding"
    model.summary()