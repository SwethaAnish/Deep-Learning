
#Import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#read and display the input image.
inp_image = cv2.imread("D:/Work/Learning/DeepLearning/tiger.jpg")
#inp_image = cv2.imread("C:/Users/F85SU00/Downloads/sample_image.jpg")
image = cv2.cvtColor(inp_image, cv2.COLOR_BGR2GRAY) 
#image=cv2.resize(image,(image.shape[0],image.shape[0]))
cv2.imshow("input image", image)
cv2.waitKey(0)
print(image.shape)

#to sharpen an image, multiply the sharpen matrix with the input image
sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]])
#to blur an image, multiply the blur matrix with the input image
blur = np.array([
    [0.0625, 0.125, 0.0625],
    [0.125,  0.25,  0.125],
    [0.0625, 0.125, 0.0625]])
#to get outline of an image, multiply the outline matrix with the input image
outline = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]])


'''when convolution is applied to any image, the input image dimensions will change based on the filter used for convolution.
This function will compute the dimensions of the convolved image'''
def convolved_image_dimension(image, kernel):
    final_height = 0
    final_width =0
    for i in range(image.shape[0]):
        if (i + kernel.shape[0] <= image.shape[0]):
            final_height+=1
    for j in range(image.shape[1]):
        if (j + kernel.shape[0] <= image.shape[1]):
            final_width+=1
    print(final_height,final_width)
    return [final_height,final_width]
convolved_image_dimension(image, outline)

#convolution is applied on the input image (outline is considered as the feature matrix)
def convolution(image, kernel):
    img_dim = convolved_image_dimension(image, outline)
    convolved_img = np.zeros(shape=(img_dim[0], img_dim[1]))
    k = kernel.shape[0]
    print(convolved_img.shape[0])
    print(convolved_img.shape[1])
    for i in range(convolved_img.shape[0]):
        for j in range(convolved_img.shape[1]):
            try:
                crop = image[i:i+k, j:j+k]
                convolved_img[i,j] = np.sum(np.multiply(crop, kernel))
            except ValueError:
                pass
    cv2.imshow("feature map", convolved_img)
    cv2.waitKey(0)
    return convolved_img

#To restore the original image dimensions, padding is done.

def get_padding_width_per_side(kernel_size):
    return kernel_size // 2
def add_padding_to_image(image, padding_width):
    img_with_padding = np.zeros(shape=(image.shape[0] + padding_width * 2,image.shape[1] + padding_width * 2))
    img_with_padding[padding_width:-padding_width, padding_width:-padding_width] = image
    return img_with_padding
pad_3x3 = get_padding_width_per_side(kernel_size=3)
pad_5x5 = get_padding_width_per_side(kernel_size=5)
convolved_image = convolution(image, sharpen)
img_with_padding_3x3 = add_padding_to_image(convolved_image, padding_width=pad_5x5)
cv2.imshow("image after padding is done", img_with_padding_3x3)
cv2.waitKey(0)


#the activation functions are implemented to incorporate non linearity in the model
def relu(x):
    return(np.maximum(0, x))
def sigmoid(x):
    return 1/(1+np.exp(-x))
def linear_function(x):
    return 9*x
def tanh_function(x):
    return (2/(1 + np.exp(-2*x))) -1
def leaky_relu_function(x):
    x_ = np.array(x.shape)
    x_ = np.where(x<=0,0.01*x,x)
    return x_
def swish_function(x):
    return  (x/(1-np.exp(-x)))
def softmax_function(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_

#activation functions
relu_image = relu(img_with_padding_3x3)
leaky_relu_image = leaky_relu_function(img_with_padding_3x3)
sigmoid_image = sigmoid(img_with_padding_3x3)
linear_function_image = linear_function(img_with_padding_3x3)
tanh_image = tanh_function(img_with_padding_3x3)
softmax_image = softmax_function(img_with_padding_3x3)
#swish_function_image = swish_function(img_with_padding_3x3)
plt.imshow(leaky_relu_image)
plt.title("leaky relu Image")
plt.show()
print(relu_image.shape)

#pooling layer 
def pooling_layer(relu_image, stride):
    pl_height = int(relu_image.shape[0]/2)
    pl_width = int(relu_image.shape[1]/2)
    image_after_avg_pooling=np.zeros(shape=(pl_height, pl_width))
    image_after_max_pooling=np.zeros(shape=(pl_height, pl_width))
    for i in range(0,relu_image.shape[0],stride):
        for j in range(0,relu_image.shape[1], stride):
            #average pooling and max pooling
            try:
                patch = relu_image[i:i+2, j:j+2]
                image_after_avg_pooling[i,j] = np.mean(patch)
                image_after_max_pooling[i,j] = np.amax(patch)
            except IndexError:
                pass
    cv2.imshow("image after avg pooling", image_after_avg_pooling)  
    cv2.imshow("image after max pooling", image_after_max_pooling)
    cv2.waitKey(0)
pooling_layer(relu_image, 2)





