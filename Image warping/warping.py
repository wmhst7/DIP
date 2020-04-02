import numpy as np
from skimage import io


def build_affine(h, w, m):
    hs = []
    hs.append(np.dot(m[1], [0, 0, 1]) / np.dot(m[2], [0, 0, 1]))
    hs.append(np.dot(m[1], [w, 0, 1]) / np.dot(m[2], [w, 0, 1]))
    hs.append(np.dot(m[1], [0, h, 1]) / np.dot(m[2], [0, h, 1]))
    hs.append(np.dot(m[1], [w, h, 1]) / np.dot(m[2], [w, h, 1]))
    ws = []
    ws.append(np.dot(m[0], [0, 0, 1]) / np.dot(m[2], [0, 0, 1]))
    ws.append(np.dot(m[0], [w, 0, 1]) / np.dot(m[2], [w, 0, 1]))
    ws.append(np.dot(m[0], [0, h, 1]) / np.dot(m[2], [0, h, 1]))
    ws.append(np.dot(m[0], [w, h, 1]) / np.dot(m[2], [w, h, 1]))
    return np.zeros((int(max(hs)), int(max(ws)), 3))


def affine(img, mat):
    print('shape:', np.shape(img))
    h = np.shape(img)[0]
    w = np.shape(img)[1]
    minv = np.linalg.inv(mat)
    img_aff = build_affine(h, w, mat)
    print('shape after affine transformation:', np.shape(img_aff))
    for x_ in range(np.shape(img_aff)[1]):
        for y_ in range(np.shape(img_aff)[0]):
            w_ = 1 - (minv[2][0]*x_ + minv[2][1]*y_) / minv[2][2]
            x = int(np.dot(minv[0], [x_*w_, y_*w_, w_]))
            y = int(np.dot(minv[1], [x_*w_, y_*w_, w_]))
            if x < w and y < h and x > -1 and y > -1:
                img_aff[y_, x_, :] = img[y, x, :]
    return img_aff


def sphere(img, arg=1.0):
    pi = 3.1415
    width = np.shape(img)[1]
    height = np.shape(img)[0]
    d0 = min(width, height)/2
    h0 = int(d0 * arg)
    img_sph = np.zeros((2 * h0, 2 * h0, np.shape(img)[2]))
    print('shape:', np.shape(img))
    print('shape_sphere:', np.shape(img_sph))
    for x_ in [int(i - h0) for i in range(2 * h0)]:
        for y_ in [int(i - h0) for i in range(2 * h0)]:
            d_ = np.sqrt(x_ * x_ + y_ * y_)
            if d_ >= h0:
                continue
            elif d_ == 0:
                x = int(width / 2.0)
                y = int(height / 2.0)
            else:
                d = 2.0 / pi * d0 * np.arcsin(d_ / h0)
                x = int(x_ * d / d_ + width / 2.0)
                y = int(y_ * d / d_ + height / 2.0)
            img_sph[y_ + h0, x_ + h0, :] = img[y, x, :]
    return img_sph


def warping_homework(img):
    io.imsave('./image/warping_homework.png', sphere(img, 1.0))


def affine_homework(imgs, imgt):
    imgs = imgs[163:420, 169:594, :]
    print('source shape:', np.shape(imgs))
    print('target shape:', np.shape(imgt))
    mat_shift = [[0.83, 0, 229],
            [0, 0.48, 150],
            [0, 0, 1]]
    mat_revolve = [[0.9784, -0.2064, 0],
            [0.2064, 0.9784, 0],
            [0, 0, 1]]
    mat = np.dot(mat_revolve, mat_shift)
    h = np.shape(imgs)[0]
    w = np.shape(imgs)[1]
    minv = np.linalg.inv(mat)
    for x_ in range(np.shape(imgt)[1]):
        for y_ in range(np.shape(imgt)[0]):
            w_ = 1 - (minv[2][0] * x_ + minv[2][1] * y_) / minv[2][2]
            x = int(np.dot(minv[0], [x_ * w_, y_ * w_, w_]))
            y = int(np.dot(minv[1], [x_ * w_, y_ * w_, w_]))
            if x < w and y < h and x > -1 and y > -1:
                imgt[y_, x_, :] = imgs[y, x, :]
    io.imsave('./image/affine_homework.jpg', imgt)


def homework():
    f1 = './image/source.jpg'
    f2 = './image/target.jpg'
    f3 = './image/warping.png'
    img1 = io.imread(f1)
    img2 = io.imread(f2)
    img3 = io.imread(f3)
    affine_homework(img1, img2)
    warping_homework(img3)
    print('Done!')


def homework1(img):
    mat = [[50, 0, 1],
           [0, 60, 1],
           [0.01, 0.02, 100]]
    return affine(img, mat)


def mywarping(img):
    width = np.shape(img)[1]
    height = np.shape(img)[0]
    imgt = np.zeros((height, width, np.shape(img)[2]))
    for x_ in range(width):
        for y_ in range(height):
            # x = int(x_ + 20 * np.cos(x_ / 20))
            # x = int(np.sin(3.14 * x_ / width) * x_ / 2 + x_ /2 )
            y = int(y_ + 2 * np.sin((x_) / 5))
            x = int(x_ + 2 * np.sin((y_) / 5))
            # y = y_
            if y < height and x < width:
                imgt[y_, x_, :] = img[y, x, :]
    return imgt


def main():
    mat = [[1.6, 0.5, 300],
           [-0.5, 1.5, 400],
           [0, 0, 1]]
    fname = './Pictures/lxydd.jpeg'
    img = io.imread(fname)
    # io.imsave(fname.replace('.jpg', '_affine.jpg'), affine(img, mat))
    # io.imsave(fname.replace('.png', '_warping.png'), mywarping(img))
    io.imsave(fname.replace('.jpeg', '_warping.jpeg'), sphere(img, 2))

    # io.imsave(fname.replace('.jpg', '_sphere.jpg'), sphere(img, 3.0))
    # io.imsave(fname.replace('.jpg', '_h.jpg'), homework1(img))

    print('Success!')
    return


main()
