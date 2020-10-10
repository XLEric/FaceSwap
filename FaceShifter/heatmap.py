import cv2
import numpy as np

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
      return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
      heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
      g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
      np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def get_heatmap(img_, output,radius=6,img_size = 256,gaussian_op = False):
    draw_gaussian = draw_msra_gaussian if gaussian_op else draw_umich_gaussian
    hm = np.zeros((img_size,img_size), dtype=np.float32)

    dict_landmarks = {}
    for i in range(int(output.shape[0]/2)):
        if 32>= i >=0:

            x = output[i*2+0]*float(img_size)
            y = output[i*2+1]*float(img_size)
            # print('x,y',x,y)

            ct = np.array([x,y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(hm, ct_int, radius)
            # cv2.circle(img_, (int(x),int(y)), 3, (0,255,0),-1)
            if 1<=x<=(img_size-2) and 1<=y<=(img_size-2):
                img_[(int(y)-1):(int(y)+1),(int(x)-1):(int(x)+1),:]=(0,255,0)

    return hm
