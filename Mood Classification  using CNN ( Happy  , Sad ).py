# ## Mood classfication using CNN (HAPPY / SAD)

# importing the necessary libraries 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

img = image.load_img(r"C:\Users\srava\Downloads\portrait-photo-japanese-infant-female-curly-hair-smiling_662214-130852.jpg")
plt.imshow(img)

i1 = cv2.imread(r"C:\Users\srava\Downloads\portrait-photo-japanese-infant-female-curly-hair-smiling_662214-130852.jpg")
i1
i1.shape # to know thw sahpe of the image 

#rescaling the Training and validation Images
train = ImageDataGenerator(rescale = 1/255)
validataion = ImageDataGenerator(rescale = 1/255)
#loading the Training and validation datset .
train_dataset = train.flow_from_directory(r"C:\Users\srava\OneDrive\Desktop\CNN -Image classification - Happy face or sad face\Training",
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')
validataion_dataset = validataion.flow_from_directory(r"C:\Users\srava\OneDrive\Desktop\CNN -Image classification - Happy face or sad face\validation",
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')
train_dataset.class_indices
train_dataset.classes

# Model Building with  maxpooling 
model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2), #3 filtr we applied hear
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),    
                                    #                       
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2), 
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    #
                                    tf.keras.layers.Dense(1,activation= 'sigmoid')
                                    ]
                                    )


#Compilation of the Model 
model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001),
              metrics = ['accuracy']
              )
#Fitting  the model with training data 
model_fit = model.fit(train_dataset,
                     steps_per_epoch = 3,
                     epochs = 15,
                     validation_data = validataion_dataset)
# testing the data 
dir_path = r"C:\Users\srava\OneDrive\Desktop\CNN -Image classification - Happy face or sad face\Testing"
for i in os.listdir(dir_path ):
    print(i)

dir_path = r"C:\Users\srava\OneDrive\Desktop\CNN -Image classification - Happy face or sad face\Testing"
for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()

dir_path = r"C:\Users\srava\OneDrive\Desktop\CNN -Image classification - Happy face or sad face\Testing"
for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+ '//'+i, target_size = (200,200))
    plt.imshow(img)
    plt.show()
    
    
    x= image.img_to_array(img)
    x=np.expand_dims(x,axis = 0)
    images = np.vstack([x])
    
    val = model.predict(images)
    if val == 0:
        print( 'i am happy')
    else:
        print('i am sad')


# Conclusion: 
# Model Predicted Exactly with 100 % Accuracy of happy and sad images at epochs 15 and epoch for step = 3 

