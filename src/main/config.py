# import sys
# sys.path.append('..')

# # 可以通过更改选择config_e2e里包含的不同参数组合
# from configs.config_e2e.config_main import *
# if config_dict["dataset"] == "XMEMRs":
#     from configs.config_e2e.config_main2 import *


# class hyperparams:
#     def __init__(self, config):
#         for key, value in config.items():
#             setattr(self, key, value)

# # Instantiate the class with the config_dict
# args = hyperparams(config_dict)
# # print(f"当前运行程序参数：{args}")

import sys
import os
sys.path.append('..')

# 获取当前运行的脚本名称，用于确定使用哪个配置
current_script = os.path.basename(sys.argv[0])
script_name = os.path.splitext(current_script)[0]

# 根据运行的脚本名称决定加载哪个配置文件
if script_name == 'main2':
    # 如果运行的是 main2.py，则加载 config_main2.py
    from configs.config_e2e.config_main2 import config_dict
    print("当前运行程序为：main2.py")
    print("配置文件为：config_main2.py")
else:
    # 否则默认加载 config_main.py
    from configs.config_e2e.config_main import config_dict
    print("当前运行程序为：main.py")
    print("配置文件为：config_main.py")


class hyperparams:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)

# Instantiate the class with the config_dict
args = hyperparams(config_dict)
# print(f"当前运行程序参数：{args}")