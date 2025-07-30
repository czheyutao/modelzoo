# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseRetriever
from . import image2image
from .image2image import ImageToImageRetriever


__all__ = ['BaseRetriever', 'ImageToImageRetriever']
