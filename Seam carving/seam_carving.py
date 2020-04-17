from skimage import io
import numpy as np
from scipy.ndimage.filters import convolve


count = 1


def get_energy(img, r, c):
    # calculate energy(too slow)
    # energy = np.zeros((r, c)).astype(int)
    # for k in range(3):
    #     for i in range(r):
    #         for j in range(c):
    #             gy = img[i + 1, j, k] - img[i - 1, j, k]
    #             gx = img[i, j + 1, k] - img[i, j - 1, k]
    #             energy[i, j] += abs(gx)
    #             energy[i, j] += abs(gy)

    # use filter to accelerate
    fri = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    fci = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    fr, fc = [], []
    for i in range(3):
        fr.append(fri)
        fc.append(fci)
    convolved = np.absolute(convolve(img, fc)) + np.absolute(convolve(img, fr))
    energy = convolved.sum(axis=2)
    return energy


def get_back(energy, r, c):
    # DP back-track
    back = np.zeros((r, c), dtype=int)
    for i in range(1, r):
        for j in range(c):
            if j == 0:
                index = np.argmin(energy[i - 1, j: j + 2])
                back[i, j] = j + index
                energy[i, j] += energy[i - 1, j + index]
            elif j == c - 1:
                index = np.argmin(energy[i - 1, j - 1: j + 1])
                back[i, j] = j - 1 + index
                energy[i, j] += energy[i - 1, j - 1 + index]
            else:
                index = np.argmin(energy[i - 1, j - 1: j + 2])
                back[i, j] = j - 1 + index
                energy[i, j] += energy[i - 1, j - 1 + index]
    return back


def carve_column(img):
    global count
    print('carve_column:', count)
    count += 1
    r = np.shape(img)[0]
    c = np.shape(img)[1]
    energy = get_energy(img, r, c)
    back = get_back(energy, r, c)
    j = np.argmin(energy[-1])
    delmask = np.ones((r, c), dtype=bool)
    for i in range(r):
        delmask[r - 1 - i, j] = False
        j = back[r - 1 - i, j]
    img = img[delmask].reshape((r, c - 1, 3))
    return img


def carve_column_obj(img, mask = None):
    r = np.shape(img)[0]
    c = np.shape(img)[1]
    energy = get_energy(img, r, c)
    # Object Removal: change energy map
    energy[np.where(mask > 0)] *= -1000.0
    back = get_back(energy, r, c)
    j = np.argmin(energy[-1])
    delmask = np.ones((r, c), dtype=bool)
    for i in range(r):
        delmask[r - 1 - i, j] = False
        j = back[r - 1 - i, j]
    img = img[delmask].reshape((r, c - 1, 3))
    mask = mask[delmask].reshape((r, c - 1))
    return img, mask


def aspect_ratio(img, newr, newc):
    img = np.copy(img).astype('float32')
    r = np.shape(img)[0]
    c = np.shape(img)[1]
    # 处理多剪除的方向
    delta = c - newc - r + newr
    num = 0
    if delta > 0:
        num = r - newr
        for i in range(delta):
            img = carve_column(img)
    else:
        num = c - newc
        img = img.transpose(1, 0, 2)
        for i in range(-delta):
            img = carve_column(img)
        img = img.transpose(1, 0, 2)
    # 交替剪除
    for i in range(num):
        img = carve_column(img)
        img = img.transpose(1, 0, 2)
        img = carve_column(img)
        img = img.transpose(1, 0, 2)
    return img


def object_removal(img, mask):
    img = img.astype('float32')
    while len(np.where(mask > 0)[0]) > 0:
        print('to bo removed:', len(np.where(mask > 0)[0]))
        img, mask = carve_column_obj(img, mask)
    return img


if __name__ == '__main__':
    # aspect ratio adjust
    # img = io.imread('./4.jpg')
    # r = np.shape(img)[0]
    # c = np.shape(img)[1]
    # newr = r
    # newc = c - 100
    # io.imsave('./aspect_ratio.jpg', aspect_ratio(img, newr, newc))

    # object removal
    img = io.imread('./4.jpg')
    mask = io.imread('./mask.jpg')
    io.imsave('./remove_object.jpg', object_removal(img, mask))




