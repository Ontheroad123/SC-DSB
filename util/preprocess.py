import cv2
import numpy as np

def clahe_process(image, clip_limit=2.0, tile_size=(16, 16)):
  """
  使用 CLAHE 对图像进行对比度增强。

  参数：
      image: 输入图像，numpy 数组格式。
      clip_limit: CLAHE 的裁剪限制，用于限制直方图均衡化的程度，默认值为 2.0。
      tile_size: CLAHE 的网格大小，用于将图像分割成子区域，默认值为 (8, 8)。

  返回值：
      处理后的图像，numpy 数组格式。
  """

  # 创建 CLAHE 对象
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

  # 将图像转换为灰度图像
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # 应用 CLAHE
  clahe_image = clahe.apply(gray)

  return clahe_image

# 加载图像
image = cv2.imread('/data/hq/diffusion/SDSB-main/dataset/ct-us/us-test/00043355_20230905_0001_000003_1.2.156.600734.0007994714.20230905.1170140.3.png')

# 使用 CLAHE 处理图像
clahe_image = clahe_process(image)

# 显示处理后的图像
cv2.imwrite('CLAHE_Image.png', clahe_image)
