
import numpy as np

def img2block(im):
    '''
    Image patching code. It patches a given RGB image into 32x32 blocks and returns a 4D array with size
    [number_of_patches,32,32,3]
    '''
    im = im.astype(np.float32)
    row, col, color = im.shape
    im_bl = np.zeros((int(row * col / 1024), 32, 32, 3)).astype(np.float32)
    count = 0
    for i in range(0, row - row % 32, 32):
        for j in range(0, col - col % 32, 32):
            im_bl[count, :, :, :] = im[i:i + 32, j:j + 32, :]
            count = count + 1
    im_bl = im_bl / 255.
    return im_bl


def block2img(img_blocks, img_size):
    '''
    Function for reconstructing the image back from patches
    '''
    row, col = img_size
    img = np.zeros((row, col, 3)).astype(np.float32)
    n, k, l, c = img_blocks.shape

    for i in range(0, int(row / k)):
        for j in range(0, int(col / k)):
            img[i * k:(i + 1) * k, j * l:(j + 1) * l, :] = img_blocks[int(i * col / k + j), :, :, :]
    return img


def convert2uint8(img):
    img[img>255]=255
    img[img<0]=0
    return img.astype(np.uint8)