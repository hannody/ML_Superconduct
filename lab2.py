from skimage.io import imread, imshow
from skimage import io
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


fname = 'mohanad.png'

img = imread(fname)

imgc = color.rgba2rgb(img)

#image_rescaled = rescale(imgc, 4.0 / 1.0, anti_aliasing=True)

image_resized = resize(
    imgc, (128, 128), anti_aliasing=True)

print(img.shape)
print(imgc.shape)
# print(image_rescaled.shape)
print(image_resized.shape)
imshow(imgc)
io.show()


# imshow(image_rescaled)
# io.show()


imshow(image_resized)
io.show()
