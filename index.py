import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop

dataset = "chihuahua-muffin/"

chihuahua_images = os.path.join('chihuahua-muffin/Chihuahua/')
muffin_images = os.path.join('chihuahua-muffin/Muffin')

print("Number of Chihuahua images: ", len(os.listdir(chihuahua_images)))
print("Number of Muffin Images: ", len(os.listdir(muffin_images)))

chihuahua_files = os.listdir(chihuahua_images)
muffin_files = os.listdir(muffin_images)

pic_index = 2

next_chihuahua = [os.path.join(chihuahua_images, fname) 
                for fname in chihuahua_files[pic_index-2:pic_index]]

next_muffin = [os.path.join(muffin_images, fname) 
                for fname in muffin_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_chihuahua+next_muffin):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    dataset,
    target_size=(150,150),
    class_mode='binary'
)

model = tf.keras.Sequential([
    
    #1st Conv2D Layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    #2nd Conv2D Layer
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

opt = RMSprop(lr=0.0001, decay=1e-6)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

history = model.fit_generator(
    train_generator,
    epochs=25,
    verbose=1
)

predict = ['Chihuahua', 'Muffin']

img = tf.keras.utils.load_img('chihuahua-muffin/muffin_fake.jpg', target_size=(150, 150))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images)
print(classes)
print(images)
for i in range(0,1):
    if classes[0][i] == 1:
        print(predict[i])