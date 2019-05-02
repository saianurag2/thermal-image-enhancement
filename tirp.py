import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('clouds.jpg',0)
#
# # create a CLAHE object (Arguments are optional).
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)
#
# cv2.imwrite('clahe_2.jpg',cl1)
# cv2.imshow(mat,'clahe_2.jpg')
# img_he = cv2.imread('clahe_2.jpg',0)
# cv2.imshow('clahe_2.jpg')
img = cv2.imread('tir_v1_0_8bit/trees/00000001.png',0)
# cv2.imshow("input",img)
equ = cv2.equalizeHist(img)
# res = np.hstack((equ)) #stacking images side-by-side
# cv2.imshow("hist eq",equ)
cv2.imwrite('res.jpg',equ)
dft = cv2.dft(np.float32(equ),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
mag,angl = cv2.cartToPolar(dft_shift[:,:,0],dft_shift[:,:,1])
print(angl)
print(mag)
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
