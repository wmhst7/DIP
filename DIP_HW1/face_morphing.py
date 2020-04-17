import numpy as np
from scipy.spatial import Delaunay
from skimage import io
import json
import cv2


def inside_triangle(point, p):
    x1 = point[0] - p[0][0]
    y1 = point[1] - p[0][1]
    x2 = p[1][0] - p[0][0]
    y2 = p[1][1] - p[0][1]
    x3 = p[2][0] - p[0][0]
    y3 = p[2][1] - p[0][1]
    u = (x1 * y3 - x3 * y1) * 1.0 / (x2 * y3 - x3 * y2)
    v = (x1 * y2 - x2 * y1) * 1.0 / (x3 * y2 - x2 * y3)
    if u >= 0 and v >= 0 and u + v <= 1:
        return True
    else:
        return False


# Affine the field within tris in the src to trit in the tar.
def affine_transformation(src, tris, trit):
    tris = np.float32([[tris[0][0], tris[0][1]],
                       [tris[1][0], tris[1][1]],
                       [tris[2][0], tris[2][1]]])
    trit = np.float32([[trit[0][0], trit[0][1]],
                       [trit[1][0], trit[1][1]],
                       [trit[2][0], trit[2][1]]])
    mat = cv2.getAffineTransform(tris, trit)
    tar = cv2.warpAffine(src, mat, (np.shape(src)[1], np.shape(src)[0]))
    return tar


def morph_one(src, tar, alpha, tris, trit):
    src_ = np.zeros(np.shape(src), dtype='float32')
    tar_ = np.zeros(np.shape(tar), dtype='float32')

    # Affine transformation
    h = min(np.shape(src)[0], np.shape(tar)[0])
    w = min(np.shape(src)[1], np.shape(tar)[1])
    res = np.zeros((h, w, np.shape(src)[2]), dtype='float32')
    tri = alpha * tris + (1.0 - alpha) * trit
    for i in range(len(tris)):
        print('{}/{}'.format(i, len(tris)))
        src_ = affine_transformation(src, tris[i], tri[i])
        tar_ = affine_transformation(tar, trit[i], tri[i])
        # Cross dissolve
        for x in range(w):
            for y in range(h):
                if inside_triangle([x, y], tri[i]):
                    res[y, x, :] = alpha * src_[y, x, :] + (1.0 - alpha) * tar_[y, x, :]

    # io.imsave('./face morphing/src_.png', src_)
    # io.imsave('./face morphing/tar_.png', tar_)
    return res


# Produce k images
def face_morphing(src, tar, k=1):
    # Detect face and landmarks
    with open('./face morphing/source1.json') as f:
        src_lm = json.load(fp=f)['faces'][0]['landmark']
    with open('./face morphing/target1.json') as f:
        tar_lm = json.load(fp=f)['faces'][0]['landmark']

    # Delaunay triangle
    spoints = np.array([[src_lm[item]['x'], src_lm[item]['y']] for item in src_lm])
    tpoints = np.array([[tar_lm[item]['x'], tar_lm[item]['y']] for item in tar_lm])
    dels = Delaunay(spoints)
    tris = spoints[dels.simplices]
    trit = tpoints[dels.simplices]
    # Calculate and morph
    for i in range(k):
        alpha = 1.0 - (i + 1.0) / (k + 1)
        out = morph_one(src, tar, alpha, tris, trit)
        io.imsave('./face morphing/morph_'+str(i+1)+'.png', out)


if __name__ == '__main__':
    src = io.imread('./face morphing/source1.png')
    tar = io.imread('./face morphing/target1.png')
    face_morphing(src, tar, 5)
