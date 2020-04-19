import cv2
import numpy as np
from scipy.spatial import Delaunay
from face_morphing import morph_one
from prewarp import get_prematrix


def normalize(pts):
    pts = np.array(pts)
    x, y = np.array(pts[:, 0]), np.array(pts[:, 1])
    n = len(x)
    x, y = x.reshape(n, 1), y.reshape(n, 1)
    mx, my = np.mean(x), np.mean(y)
    x_sh = x - mx
    y_sh = y - my
    c = np.sqrt(2.0) / np.mean(np.sqrt(x_sh**2 + y_sh**2))
    mat = np.array([[c, 0, -c * mx],
           [0, c, -c * my],
           [0, 0, 1]])
    one = np.ones((n, 1))
    xy = np.concatenate((x, y, one), axis=1)
    xy_norm = np.dot(mat, np.transpose(xy))
    return np.transpose(xy_norm), mat


# get fundamental matrix based on eight points algorithm
def get_camera(spts, tpts):
    sxy, smat = normalize(spts)
    txy, tmat = normalize(tpts)
    sx, sy = sxy[:, 0], sxy[:, 1]
    tx, ty = txy[:, 0], sxy[:, 1]
    n = len(sx)
    mul = lambda a, b: np.multiply(a, b)
    one = np.ones(n)
    A = np.concatenate((mul(tx, sx), mul(tx, sy), tx,
                        mul(ty, sx), mul(ty, sy), ty, sx, sy, one)).reshape((n, -1), order='F')
    U, D, V = np.linalg.svd(A)
    small = V[-1, :].T
    F = small.reshape(3, 3)
    U, D, V = np.linalg.svd(F)
    r, s = D[0], D[1]
    F = np.dot(U, np.diag([r, s, 0])).dot(V)
    F = tmat.T.dot(F).dot(smat)
    return F


def get_pre_points(H0, src_points):
    n = len(src_points)
    ones = np.ones((n, 1))
    src_points = np.concatenate((src_points, ones), axis=1)
    srcn = src_points.dot(H0)
    srcr = [[it[0] / it[2], it[1] / it[2]] for it in src_points]
    return srcr






def view_morphing(src, tar, src_points, tar_points):
    # Find the camera
    F = get_camera(src_points[8:], tar_points[8:])
    print('F:', F)

    # Get prewarp matrix
    H0, H1 = get_prematrix(F)

    # Pre-warp
    h, w = np.shape(src)[0], np.shape(src)[1]
    newh = int(np.sqrt(h * h + w * w))
    src_pre = cv2.warpPerspective(src, H0, (newh, newh))
    tar_pre = cv2.warpPerspective(tar, H1, (newh, newh))
    cv2.imwrite('view morphing/src_prewarp.png', src_pre)
    cv2.imwrite('view morphing/tar_prewarp.png', tar_pre)
    src_pre_points = get_pre_points(H0, src_points)
    tar_pre_points = get_pre_points(H1, tar_points)

    # Delaunay triangle
    dels = Delaunay(src_pre_points)
    src_tri = np.array(src_pre_points)[dels.simplices]
    tar_tri = np.array(tar_pre_points)[dels.simplices]

    # Morph the shape! need to debug!
    morph = morph_one(src_pre, tar_pre, 0.5, src_tri, tar_tri)
    cv2.imwrite('view morphing/morph.png', morph)

    # Get points pairs to determine Hs
    Hs = get_post_matrix()

    # Post-warp



    return


if __name__ == '__main__':
    number = '1'
    src = cv2.imread('view morphing/source_'+number+'.png')
    tar = cv2.imread('view morphing/target_'+number+'.png')
    with open('view morphing/source'+number+'_point.json') as f:
        src_points = eval(f.read())
    with open('view morphing/target'+number+'_point.json') as f:
        tar_points = eval(f.read())
    view_morphing(src, tar, src_points, tar_points)
