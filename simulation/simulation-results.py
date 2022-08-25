import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from astropy.io import fits
import pandas as pd


plt.plot([2, 3, 4, 5], [0.8088, 0.8395, 0.7886, 0.40], marker='o', ms=5, mfc='r')
plt.title('kernel size every layer')
plt.ylabel('accuracy')
plt.xlabel('kernel size')
plt.xticks([2, 3, 4, 5])
plt.show()

fig, ((ax1, ax2,ax3)) = plt.subplots(3,1, figsize=(10, 8))
ax1.plot([2, 3, 4, 5], [0.8088, 0.8395, 0.7886, 0.6423], marker='o', ms=5, mfc='r')
ax1.set_title('kernel size every layer')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('kernel size')
ax1.set_xticks([2, 3, 4, 5])

ax2.plot([45, 120, 240, 480], [0.5768, 0.8492, 0.8560, 0.8320], marker='o', ms=5, mfc='r')
ax2.set_title('feature maps count')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('feature maps')
ax2.set_xticks([45, 120, 240, 480])

fig.subplots_adjust(hspace=0.7, wspace=0.4)

ax3.plot([1, 5, 15, 50], [0.8795, 0.8492, 0.48, 0.20], marker='o', ms=5, mfc='r')
ax3.set_title('leaning rate x$\mathregular{10^{-3}}$')
ax3.set_ylabel('accuracy')
ax2.set_xlabel('learning rage')
ax3.set_xticks([1, 5, 15, 50])



