#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import nibabel as nib
import nrrd
import tensorflow as tf





def get_imgs_labels_paths(PATH):
    
    dataset_path = {}
    imgs_path =[]
    labls_path = []
    number_images = []

    for patient in os.listdir(PATH):
        # to check if the patient  folder name ends with a digit
        if patient.split('/')[-1].isdigit():
            
            for technic in  os.listdir(os.path.join(PATH, patient)):
                if technic in ['ADC','DWI', 'T2W']:
                    images_path = sorted(glob.glob(os.path.join(PATH, patient, technic)+'/*.nii'))
                    imgs_path.append(images_path)
                    number_images.append(len(images_path))


                elif technic in ['label']:
                    labels_path = sorted(glob.glob(os.path.join(PATH, patient, technic)+'/*.nrrd'))
                    labls_path.append(labels_path)
                    
                else:
                    print('there is another folder named: ', PATH, patient, technic)

            number_images.append(len(labels_path))

            dataset_path[os.path.join(PATH, patient)] = [imgs_path, labels_path, number_images]
            imgs_path =[]
            labls_path =[]
            number_images = []
    return dataset_path


def read_label(img_path, resize_shape , num_classes):
    readdata, header = nrrd.read(img_path)
    label = cv2.resize( readdata, resize_shape)
    label = label.astype(np.uint8)
    label = tf.one_hot(tf.squeeze(label), depth= num_classes)
    label = label.numpy().astype(np.uint8)
    #print(img_path, 'label shape', label.shape)
    return label



def read_image(path, resize_shape):
        
    image = nib.load(path) 
    image = np.array(image.dataobj)
    image = image.astype(float)
    image = cv2.resize( image, resize_shape )
    image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    #print(path, 'image shape', image.shape)
    return image



def read_imgs_labels(dataset_paths, resize_shape, num_classes):
    
    x_train = []
    y_train = []
    for k , i in dataset_paths.items():
        #print(k)
        #print()
        image_paths = i[0]
        label_path = i[1]
        number_paths = i[2]
        assert all([1 == num for num in number_paths])
        
        for a, d, t, l in zip(image_paths[0], image_paths[1], image_paths[2],label_path  ):
            img_adc = read_image(a, resize_shape)
            img_dwi = read_image(d, resize_shape)
            img_t2w = read_image(t, resize_shape)
            label_img = read_label(l, resize_shape , num_classes)
            assert img_adc.shape == img_dwi.shape == img_t2w.shape == label_img.shape[:3]

            for ch in range(img_adc.shape[2]):
                x_train.append(np.stack([img_adc[:,:,ch],img_dwi[:,:,ch],img_t2w[:,:,ch]], axis=2))
                y_train.append(label_img[:,:,ch,:])
        #print('--------------------------------------------------------------')
    x_train = np.stack((x_train))
    y_train = np.stack((y_train))
    return x_train, y_train


@tf.function()
def preparation(image, label , center_crop_rate=0.7, input_shape=(256, 256) ):
    
  
    image =  tf.image.central_crop(image, center_crop_rate)
    label =  tf.image.central_crop(label, center_crop_rate)

    image =  tf.image.resize(image, input_shape, method='bilinear')
    label =  tf.image.resize(label, input_shape, method='bilinear')
  
    image = tf.cast(image, dtype= tf.float32)
    label = tf.cast(label, dtype= tf.float32 )  
    
    return image, label

@tf.function()
def normalize(image, label):
    # normalizing the images to [-1, 1]
   
    image = tf.image.per_image_standardization(image)
    #image = (image / 127.5) - 1

    return image, label


@tf.function()
def random_augmentation(image, label):
    
        
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image, k=1, name=None)
        label = tf.image.rot90(label, k=1, name=None)
        
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image, k=3, name=None)
        label = tf.image.rot90(label, k=3, name=None)
        
    
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)

    return image, label

@tf.function()
def load_image_train(image_file, label_file, input_shape):
    image, label= preparation(image_file, label_file, center_crop_rate=0.7, input_shape=input_shape)
    image, label = random_augmentation(image, label)
    image, label = normalize(image, label)
    return image, label

@tf.function()
def load_image_test(image_file, label_file, input_shape):
    image, label= preparation(image_file, label_file, center_crop_rate=0.7, input_shape=input_shape)
    #image, label = random_augmentation(image, label)
    image, label = normalize(image, label)
    return image, label


def create_train_test_dataset(x_train, y_train, number_test_image, buffer_size, batch_size, input_shape):
    
    x_test, y_test = x_train[:number_test_image,:,:,], y_train[:number_test_image,:,:,]
    x_train, y_train = x_train[number_test_image:,:,:,], y_train[number_test_image:,:,:,]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(lambda x, y: load_image_train(x, y, input_shape) , num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.map(lambda x, y: load_image_test(x, y, input_shape) , num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset

def show_img(img,label, n_classes):
    img = img[0,:,:,:]
    label = label[0,:,:,:]

    plt.imshow(img)
    fig, axs = plt.subplots(1,n_classes, figsize=(15, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    for i in range(n_classes):
        axs[i].imshow(label[:,:,i])
        axs[i].set_title('Ground T of Channel ' + str(i))
        print('Unique numbers in channel {} are {},{}'.format(i, np.min(np.unique(label[:, :, i])),
                                                              np.max(np.unique(label[:, :, i]))))

    plt.show()