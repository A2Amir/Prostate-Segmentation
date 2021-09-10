# Prostate-Segmentation
Magnetic resonance imaging (MRI) produces detailed anatomical images of the prostate and its areas. It plays a crucial role in many diagnostic applications. Automatic segmentation of prostate and prostate zones from MR images makes many diagnostic and therapeutic applications easier. In this repository, I tried to segment prostate using Pix2Pix network.

## 1. Datasets 
Two datasets (Train and Test) were created using different the **combinations of images, including T2-Weighted (T2W) images, Diffusion-Weighted Images (DWI) and Apparent Diffusion Coefficient (ADC) images**. To merge and aligne the DWI, ADC and T2W images, I used an established registration toolbox for transformation [1].

![grafik](./imgs/1.PNG)


**Train Dataset Shape:**
* Image Shape: (669, 256, 256, 3)
* Label shape (669, 256, 256, 5)

**Test Dataset Shape:**
* Image Shape: (50, 256, 256, 3)
* Label shape (50, 256, 256, 5)

## 2. Normalization
In normalising the images, the mean value of all images is subtracted from the signal intensity of each pixel and the value obtained is divided by the standard deviation.

## 3. Data Augmentation
For the training sample, we rotated the data by 90◦ , flipped it horizontally and flipped it vertically. 

[1]: Klein, S.; Staring, M. Elastix: A toolbox for intensity-based medical image registration. IEEE Trans. Med. Imaging 2010, 29, 196–205.
[CrossRef] [PubMed]

## 3.Loss Function:

### 3.1 Generator loss

   * It is a sigmoid cross entropy loss of the generated images and an array of ones.
   * The paper also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
   * This allows the generated image to become structurally similar to the target image.
   * The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the paper.

### 3.2 Discriminator loss

   * The discriminator loss function takes 2 inputs; real images, generated images
   * real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since these are the real images)
   * generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since these are the fake images)
   * Then the total_loss is the sum of real_loss and the generated_loss

## 3. Training

The Pix2Pix model was implemented with with TensorFlow (v. 2.4) by using Python (v. 3.8). I used a Adam optimizer to update the weights with an initial learning rate of 2e-4 and a batch size of 1. Training with 300 epochs usually achieved the lowest loss and therefore was employed in my experiments. 

