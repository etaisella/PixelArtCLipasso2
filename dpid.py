import cv2
from PIL import Image
import numpy as np
import torch

def get_guidance_image(input, d):
  h, w, c = input.shape
  assert h % d == 0, f"input must be a multiple of d currently"
  assert w % d == 0, f"input must be a multiple of d currently"
  new_h = int(h / d)
  new_w = int(w / d)

  # create result image
  result_im = np.zeros((new_h, new_w, c))

  # create filters
  box = np.ones((d, d, c))*(1 / (d * d))
  gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1 / 16)

  # iterate over dxd block and perform box filter
  for i in range(0, new_h):
    for j in range(0, new_w):
      block = input[i*d:(i + 1)*d, j*d:(j + 1)*d].astype(np.float32)
      result_im[i, j] = np.sum(np.sum(block * box, axis=1), axis=0)

  # run gaussian smoothing on intermediate result
  result_im = cv2.filter2D(src=result_im, ddepth=-1, kernel=gauss)

  # return result
  result_im = result_im.astype(np.uint8)
  return result_im


def dpid(input, d, lamb=1.0):
  h, w, c = input.shape
  assert h % d == 0, f"input must be a multiple of d currently"
  assert w % d == 0, f"input must be a multiple of d currently"
  new_h = int(h / d)
  new_w = int(w / d)
  eps = 0.00000001

  # get guidance image:
  guidance_image = get_guidance_image(input, d)

  # calc vmax:
  vmax = np.sqrt(3*255*255)

  # set result image:
  result_im = np.zeros_like(guidance_image)

  # iterate over dxd block and perform DPID:
  for i in range(0, new_h):
    for j in range(0, new_w):
      block = input[i*d:(i + 1)*d, j*d:(j + 1)*d].astype(np.float32)
      guide_pix = guidance_image[i, j]

      # get distances and weights:
      distances = np.sqrt(np.sum((block - guide_pix) ** 2, axis=-1)) + eps
      distances[block[..., 3] < 128] = eps
      distances = np.repeat(np.expand_dims(distances, axis=-1), 4, axis=-1)
      weights = (distances / vmax) ** lamb

      # get kp and final value
      kp = np.sum(np.sum(weights, axis=1), axis=0)
      result_im[i, j] = np.sum(np.sum(block * weights, axis=1), axis=0) /  kp


  # return result
  result_im = result_im.astype(np.uint8)
  return result_im