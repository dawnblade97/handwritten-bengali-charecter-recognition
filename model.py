import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D,Activation
from keras.applications import VGG16
img_width, img_height = 224, 224
train_dir = "C://Users//maste//Desktop//dl_project//data1//BasicFinalDatabase//Train"
valid_dir = "C://Users//maste//Desktop//dl_project//data1//BasicFinalDatabase//Test"

def train_model():

    import os, shutil
    from keras.preprocessing.image import ImageDataGenerator
    conv_base = VGG16(weights='imagenet', 
                      include_top=False,
                      input_shape=(img_width, img_height, 3))

    # Show architecture
    conv_base.summary()
    

    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 24

    def extract_features(directory, sample_count):
        features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
        labels = np.zeros(shape=(sample_count,50))
        # Preprocess data
        generator = datagen.flow_from_directory(directory,
                                                target_size=(img_width,img_height),
                                                batch_size = batch_size,
                                                class_mode='categorical')
        # Pass data through convolutional base
        i = 0
        for inputs_batch, labels_batch in generator:
            features_batch = conv_base.predict(inputs_batch)
            features[i * batch_size: (i + 1) * batch_size] = features_batch
            labels[i * batch_size: (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= sample_count:
                break
        return features, labels


    train_features, train_labels = extract_features(train_dir, 240*50)
    validation_features, validation_labels = extract_features(valid_dir, 60*50)

    epochs = 200

    model = Sequential()
    #model.add(GlobalAveragePooling2D(input_shape=(7,7,512)))
    '''
    model.add(MaxPooling2D(input_shape=(7,7,512),pool_size=(2,2)))
    model.add(Dropout(0.17))
    model.add(Flatten())
    model.add(Dropout(0.15))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='softmax'))
    model.summary()
    '''
    model.add(MaxPooling2D(input_shape=(7,7,512),pool_size=(2,2), strides=(2,2), padding='valid'))


    model.add(Flatten())

    model.add(Dense(4096))
    model.add(Activation('relu'))

    model.add(Dropout(0.15))


    model.add(Dense(4096))
    model.add(Activation('relu'))

    model.add(Dropout(0.15))


    model.add(Dense(1000))
    model.add(Activation('relu'))

    model.add(Dropout(0.15))

    model.add(Dense(50))
    model.add(Activation('softmax'))
    model.summary()





    from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
    checkpoint = ModelCheckpoint('vgg16_mod.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  
    red=ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=5,mode="auto")
    early=EarlyStopping(monitor="val_loss",min_delta=1e-4,patience=10,mode="auto")
    # Compile model
    from keras.optimizers import Adam,SGD
    sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    # Train model
    history = model.fit(train_features, train_labels,
                        epochs=epochs,
                        batch_size=batch_size, 
                        callbacks=[checkpoint,red,early],
                        validation_data=(validation_features, validation_labels))

def test_model(img_name):
    conv_base = VGG16(weights='imagenet', 
                      include_top=False,
                      input_shape=(img_width, img_height, 3))

    # Show architecture
    conv_base.summary()
    
    model=load_model("vgg16_mod.h5")
    from keras.preprocessing import image
    import matplotlib.pyplot as plt
    label_map = {1:'aw',2:'aaa',3:'e',4:'eee',5:'u',6:'ooo',
             7:'rhi', 8:'ey',9:'oi', 10:'o', 11:'ou',
             12: 'kaw', 13:'khaw', 14:'gaw', 15:'ghaw', 16:'ng0',
             17: 'ch0', 18:'chho', 19:'jaw', 20:'jhaw', 21:'N0',
             22: 'taw', 23:'thaw', 24:'daw', 25:'dhaw', 26:'naw(1)',
             27: 'ta(w)', 28: 'tha(w)', 29:'da(w)', 30:'dha(w)', 31:'naw(2)',
             32: 'paw', 33:'faw', 34:'baw', 35:'bhaw', 36:'maw',
            37:'jaw2',38:'raw',39:'law',40:'haw', 41:'saw1',42:'saw2',43:'saw3', 
             44:'ya',45:'ra',46:'rha',47 :'khando-taw', 48:'onussar',49:'bisargo', 50:'chandrabindu'}
    def prediction(img_path):
        org_img = image.load_img(img_path)
        img = image.load_img(img_path, target_size=(img_width, img_height))
        img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
        img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application
        plt.imshow(org_img)                           
        plt.axis('off')
        plt.show()


        # Extract features
        features = conv_base.predict(img_tensor.reshape(1,img_width, img_height, 3))

        # Make prediction
        try:
            prediction = model.predict(features)
        except:
            prediction = model.predict(features.reshape(1, 7*7*512))
            
        
        #print(np.array(prediction[0]))
        print("I see..."+str(label_map[np.argmax(np.array(prediction[0]))+1]))
        return

    p="C://Users//maste//Desktop//dl_project//tl_model//samp_images"
    prediction(p+"//"+img_name)

def main():
    
    parser=argparse.ArgumentParser(description="charecter recognition")
    parser.add_argument("--train",help="train the model")
    parser.add_argument("--test",help="enter the image name")
    args=parser.parse_args()
    if args.train:
        train_model()
    elif args.test :
        iname=args.test
        test_model(iname)
    else:
        print("no arguement passed see help for details")
        return
    
    

    

    
        
        
        
main()
    
        
    
    
    




