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
    sxy, t1 = normalize(spts)
    txy, t2 = normalize(tpts)
    u1, v1, u2, v2 = txy[:, 0], txy[:, 1], sxy[:, 0], sxy[:, 1]
    n = len(u1)
    one = np.ones(n)
    A = np.array([u1 * u2, u1 * v2, u1, v1 * u2, v1 * v2, v1, u2, v2, one]).T
    U, D, V = np.linalg.svd(A)
    small = V.T[:, -1]
    F = small.reshape(3, 3)
    U, D, V = np.linalg.svd(F)
    r, s = D[0], D[1]
    F = U.dot(np.diag([r, s, 0])).dot(V)
    F = t2.T.dot(F).dot(t1)
    F = F / F[2, 2]
    return F


def get_pre_points(H0, src_points):
    n = len(src_points)
    ones = np.ones((n, 1))
    src_points = np.concatenate((src_points, ones), axis=1)
    srcn = H0.dot(src_points.T).T
    srcr = [[it[0] / it[2], it[1] / it[2]] for it in srcn]
    return srcr


def get_post_matrix(mp, pp):
    mp = np.array(mp)
    pp = np.array(pp)
    Hs, _ = cv2.findHomography(mp, pp)
    return Hs


def view_morphing(src, tar, src_points, tar_points):
    # Find the camera
    F = get_camera(src_points, tar_points)
    print('F:', F)

    # Get prewarp matrix
    H0, H1 = get_prematrix(F)

    # Pre-warp
    h, w = np.shape(src)[0], np.shape(src)[1]
    hh, ww = np.shape(tar)[0], np.shape(tar)[1]
    s_points = [[0, 0], [0, h], [w, 0], [w, h]]
    t_points = [[0, 0], [0, hh], [ww, 0], [ww, hh]]
    newh = int(np.sqrt(h * h + w * w))
    src_pre = cv2.warpPerspective(src, H0, (newh, newh))
    tar_pre = cv2.warpPerspective(tar, H1, (newh, newh))
    cv2.imwrite('view morphing/src_prewarp.png', src_pre)
    cv2.imwrite('view morphing/tar_prewarp.png', tar_pre)
    src_points_a = np.concatenate((s_points, src_points), axis=0)
    tar_points_a = np.concatenate((t_points, tar_points), axis=0)

    src_pre_points = get_pre_points(H0, src_points_a)
    tar_pre_points = get_pre_points(H1, tar_points_a)

    # Delaunay triangle
    dels = Delaunay(src_pre_points)
    src_tri = np.array(src_pre_points)[dels.simplices]
    tar_tri = np.array(tar_pre_points)[dels.simplices]


    for alpha in [0.25, 0.5, 0.75]:
        p_points = np.multiply(s_points, alpha) + np.multiply(t_points, 1 - alpha)
        m_points = np.multiply(get_pre_points(H0, s_points), alpha) + \
                   np.multiply(get_pre_points(H1, t_points), 1 - alpha)

        # Morph
        morph = morph_one(src_pre, tar_pre, alpha, src_tri, tar_tri)
        cv2.imwrite('view morphing/morph'+str(alpha)+'.png', morph)

        # Post-warp
        Hs = get_post_matrix(m_points, p_points)
        res = cv2.warpPerspective(morph, Hs, (max(w, ww), max(h, hh)))
        cv2.imwrite('view morphing/post_warp'+str(alpha)+'.png', res)
    return


if __name__ == '__main__':
    number = '3'
    src = cv2.imread('view morphing/source_'+number+'.png')
    tar = cv2.imread('view morphing/target_'+number+'.png')
    with open('view morphing/source'+number+'_point.json') as f:
        src_points = eval(f.read())
    with open('view morphing/target'+number+'_point.json') as f:
        tar_points = eval(f.read())
    view_morphing(src, tar, src_points, tar_points)
