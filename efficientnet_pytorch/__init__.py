__version__ = "0.7.1"

# 从 model.py 导入主要类
from .model import EfficientNet, VALID_MODELS, MBConvBlock

# 从 utils.py 导入辅助函数和类
from .utils import (
    # 参数和配置类
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    
    # 模型构建辅助函数
    efficientnet,
    get_model_params,
    efficientnet_params,
    
    # 网络层相关
    SwishImplementation,
    MemoryEfficientSwish,
    Conv2dDynamicSamePadding,
    Conv2dStaticSamePadding,
    MaxPool2dDynamicSamePadding,
    MaxPool2dStaticSamePadding,
    
    # 实用工具函数
    round_filters,
    round_repeats,
    drop_connect,
    get_width_and_height_from_size,
    calculate_output_image_size,
    get_same_padding_conv2d,
    get_same_padding_maxPool2d,
)
