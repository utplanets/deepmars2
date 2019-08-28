import numpy as np
import numba

def _window_sum_2d(image, window_shape):

    window_sum = np.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                  - window_sum[:-window_shape[0] - 1])

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1]
                  - window_sum[:, :-window_shape[1] - 1])
    
    return window_sum

@numba.jit
def conv(img, ker):
    ix,iy = np.where(img == 1)
    kx,ky = np.where(ker == 1)
    conv_dim = img.shape[0] + ker.shape[0]
    conv = np.zeros((conv_dim, conv_dim))
    for i in range(len(kx)):
        for j in range(len(ix)):
            conv[kx[i]+ix[j],ky[i]+iy[j]] +=1
    return conv[ker.shape[0]-1:-ker.shape[0], ker.shape[1]-1:-ker.shape[1]]

def match_template(image, template, pad_input=False):
    image_shape = image.shape
    pad_width = tuple((width, width) for width in template.shape)
    image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)

    image_window_sum = _window_sum_2d(image, template.shape)
    image_window_sum2 = _window_sum_2d(image ** 2, template.shape)

    template_mean = template.mean()
    template_volume = np.prod(template.shape)
    template_ssd = np.sum((template - template_mean) ** 2)

    #xcorr = fftconvolve(image, template[::-1, ::-1], mode="valid")[1:-1, 1:-1]
    xcorr = conv(image, template[::-1, ::-1])[1:-1, 1:-1]
    
    numerator = xcorr - image_window_sum * template_mean
    denominator = image_window_sum2
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    np.divide(image_window_sum, template_volume, out=image_window_sum)
    denominator -= image_window_sum
    denominator *= template_ssd
    np.maximum(denominator, 0, out=denominator)
    np.sqrt(denominator, out=denominator)

    response = np.zeros_like(xcorr, dtype=np.float64)
    mask = denominator > np.finfo(np.float64).eps
    response[mask] = numerator[mask] / denominator[mask]

    slices = []
    for i in range(template.ndim):
        if pad_input:
            d0 = (template.shape[i] - 1) // 2
            d1 = d0 + image_shape[i]
        else:
            d0 = template.shape[i] - 1
            d1 = d0 + image_shape[i] - template.shape[i] + 1
        slices.append(slice(d0, d1))
    return response[tuple(slices)]

@numba.jit
def my_window_sum(A, ker_dim):
    A = A[1:,1:]
    A_dim = A.shape[0]
    W_dim = A_dim - ker_dim
    W_tmp = np.zeros((A_dim, W_dim))
    W_tmp[:,0] = np.sum(A[:,:ker_dim], axis=1)
    for i in range(1,W_dim):
        W_tmp[:,i] = W_tmp[:,i-1] + A[:,i+ker_dim-1] - A[:,i-1]
    W = np.zeros((W_dim, W_dim))
    W[0,:] = np.sum(W_tmp[:ker_dim,:], axis=0)
    for j in range(1,W_dim):
        W[j,:] = W[j-1,:] + W_tmp[j+ker_dim-1,:] - W_tmp[j-1,:]

    return W

@numba.jit
def my_match_template_2(img, ker, pad_input=False):
    img_dim = img.shape[0]
    ker_dim = ker.shape[0]
    
    pad_dim = img_dim + 2 * ker_dim
    img_pad = np.zeros((pad_dim, pad_dim))
    img_pad[ker_dim:-ker_dim,ker_dim:-ker_dim] = img
    
    img_ws = my_window_sum(img_pad, ker_dim)
    img2_ws = my_window_sum(img_pad ** 2, ker_dim)

    ker_mean = ker.mean()
    ker_vol = ker_dim **2
    ker_ssd = np.sum((ker - ker_mean) ** 2)

    xcorr = conv(img_pad, ker[::-1, ::-1])[1:-1, 1:-1]
    
    numerator = xcorr - img_ws * ker_mean
    denominator = img2_ws
    img_ws = img_ws **2
    img_ws = img_ws / ker_vol
    denominator -= img_ws
    denominator *= ker_ssd
    denominator = np.maximum(denominator, np.finfo(np.float64).eps)
    denominator = np.sqrt(denominator)
    
    response = numerator / denominator
    
    d0 = (ker_dim - 1) // 2
    d1 = d0 + img_dim
    return response[d0:d1,d0:d1]