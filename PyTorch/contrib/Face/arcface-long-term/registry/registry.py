# mmpretrain/registry.py

from mmengine.registry import Registry

# 这里注册器名字要跟代码中调用的保持一致
MODELS = Registry('model')
