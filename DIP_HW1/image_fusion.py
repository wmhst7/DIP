from skimage import io
import numpy as np
from scipy.sparse import linalg
from scipy.sparse import lil_matrix


# loc: (shift in row, shift in col)
def shift(src, tar, mask, loc):
    row = np.shape(tar)[0]
    col = np.shape(tar)[1]
    newsrc = np.zeros(np.shape(tar)).astype(int)
    newmask = np.zeros((row, col)).astype(int)
    for r in range(np.shape(src)[0]):
        for c in range(np.shape(src)[1]):
            if r + loc[0] < row and c + loc[1] < col:
                if r + loc[0] >= 0 and c + loc[1] >= 0:
                    newsrc[r + loc[0], c + loc[1]] = src[r, c]
                    if mask[r, c] > 0:
                        newmask[r + loc[0], c + loc[1]] = 1
    return newsrc, newmask


def inside(i, j, mask):
    if mask[i, j] != 0:
        return True
    else:
        return False


def get_points(mask):
    points = []
    for i in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[1]):
            if inside(i, j, mask):
                points.append((i, j))
    return points


def get_neighbors(point):
    i, j = point[0], point[1]
    return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]


def get_A(points, n):
    a = lil_matrix((n, n))
    for i in range(n):
        a[i, i] = 4
        for neighbor in get_neighbors(points[i]):
            if neighbor in points:
                j = points.index(neighbor)
                a[i, j] = -1
    return a


def lapl(point, src):
    lap = 4 * src[point]
    for n in get_neighbors(point):
        lap -= src[n]
    return lap


def out_border_neighbors(point, mask):
    outn = []
    for nb in get_neighbors(point):
        if not inside(nb[0], nb[1], mask):
            outn.append(nb)
    return outn


def get_b(points, src, tar, mask):
    n = len(points)
    b = np.zeros(n)
    for i in range(n):
        b[i] = lapl(points[i], src)
        for op in out_border_neighbors(points[i], mask):
            b[i] += tar[op]
    return b


def poisson_fusion(src, tar, mask, loc):
    # 将src图片移到合适的位置
    src, mask = shift(src, tar, mask, loc)

    # 得到需要填充的点
    points = get_points(mask)
    # print(points)

    # 构造A
    N = len(points)
    print('Points:', N)
    A = get_A(points, N).asformat('csr')

    # 构造b
    b = get_b(points, src, tar, mask)

    # 解方程并填充
    # print(len(b), )
    x = linalg.spsolve(A, b)
    # print(A, b, x)
    res = np.copy(tar).astype(int)
    for index in range(N):
        i, j = points[index][0], points[index][1]
        res[i, j] = int(x[index])
    return res


if __name__ == '__main__':
    src = io.imread('./image fusion/src3.jpg')
    tar = io.imread('./image fusion/target3.jpeg')
    mask = io.imread('./image fusion/mask3.jpg')
    channel = np.shape(tar)[2]
    loc = (160, 600)
    res = np.zeros(np.shape(tar))
    print('channel:', channel)
    for i in range(channel):
        res[:, :, i] = poisson_fusion(src[:, :, i], tar[:, :, i], mask[:, :, 0], loc)
    io.imsave('./image fusion/result3.jpg', res)
