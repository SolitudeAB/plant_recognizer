import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json
import time
import io
import pickle
from typing import List, Tuple

# ==============================================================================
# 🚀 配置和设备设置
# ==============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"-> 检测到设备: {device}")

# Pl@ntNet-300K 配置
NUM_CLASSES = 1081 
WEIGHTS_PATH = 'data.pkl' 
IMAGE_PATH = 'test_plant.jpg' 
SPECIES_NAME_JSON = 'plantnet300K_species_id_2_name.json'
CLASS_TO_ID_JSON = 'class_idx_to_species_id.json' 

# ==============================================================================
# 辅助函数：加载类别名称
# ==============================================================================
def load_class_names(class_to_id_path: str, species_name_path: str, num_classes: int) -> List[str]:
    print("-> 正在加载 1081 个植物类别名称...")
    if not os.path.exists(class_to_id_path) or not os.path.exists(species_name_path):
        raise FileNotFoundError(f"缺少必要的 JSON 文件。")
    try:
        with open(class_to_id_path, 'r', encoding='utf-8') as f:
            class_to_id = json.load(f)
        with open(species_name_path, 'r', encoding='utf-8') as f:
            species_id_to_name = json.load(f)
        class_names = []
        for class_index in range(num_classes):
            species_id = str(class_to_id[str(class_index)])
            species_name = species_id_to_name.get(species_id, f"Unknown Species ID {species_id}")
            class_names.append(species_name)
        print("-> 类别名称加载成功。")
        return class_names
    except Exception as e:
        print(f"加载类别名称时发生错误：{e}")
        raise

# ==============================================================================
# 核心函数：模型加载 (Unwrap Checkpoint)
# ==============================================================================
def load_plant_model(num_classes: int, weights_path: str) -> nn.Module:
    print(f"-> 正在加载预训练的 ResNet-18 模型...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) 
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"缺少权重文件：{weights_path}。")
        
    print(f"-> 发现权重文件: {weights_path}。正在加载 Checkpoint...")
    
    # 检查 data 文件夹
    base_dir = os.path.dirname(os.path.abspath(weights_path))
    data_dir = os.path.join(base_dir, 'data')
    
    # --- 自定义 Unpickler 类 ---
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

        def persistent_load(self, saved_id):
            if isinstance(saved_id, tuple) and saved_id[0] == 'storage':
                typename, key, location, numel = saved_id[1], saved_id[2], saved_id[3], saved_id[4]
                if isinstance(typename, type):
                    typename_str = typename.__name__
                else:
                    typename_str = str(typename)
                
                storage_cls = torch.FloatStorage
                if 'FloatStorage' in typename_str: storage_cls = torch.FloatStorage
                elif 'LongStorage' in typename_str: storage_cls = torch.LongStorage
                elif 'IntStorage' in typename_str: storage_cls = torch.IntStorage
                elif 'DoubleStorage' in typename_str: storage_cls = torch.DoubleStorage
                elif 'HalfStorage' in typename_str: storage_cls = torch.HalfStorage
                elif 'ByteStorage' in typename_str: storage_cls = torch.ByteStorage
                elif 'BoolStorage' in typename_str: storage_cls = torch.BoolStorage

                data_file_path = os.path.join(data_dir, str(key))
                if not os.path.exists(data_file_path):
                     # 如果找不到 data 文件夹，为了避免在这里崩溃，尝试返回 None 或者抛出更清晰的错误
                     # 但通常前面的检查已经覆盖了。这里我们假设路径正确。
                     pass
                
                return storage_cls.from_file(data_file_path, shared=False, size=numel)
            return saved_id

    try:
        with open(weights_path, 'rb') as f:
            # 1. 加载整个 Checkpoint 字典
            checkpoint = CustomUnpickler(f).load()
        
        # 2. [核心修复] 提取真正的权重字典
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print("-> 检测到 Checkpoint 格式，正在提取 'model' 键...")
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 3. 加载权重到模型
        model.load_state_dict(state_dict)
        print("-> 权重加载成功！")
        
    except Exception as e:
        print(f"-> [加载错误] 详细信息: {e}")
        raise e
    
    model = model.to(device)
    model.eval()
    return model

# ==============================================================================
# 核心函数：图像预处理与推理
# ==============================================================================
def preprocess_image(image_path: str) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到图像文件: {image_path}")
        
    preprocess = transforms.Compose([
        transforms.Resize(256),            
        transforms.CenterCrop(224),        
        transforms.ToTensor(),             
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0).to(device)

def classify_plant(model: nn.Module, input_tensor: torch.Tensor, class_names: List[str]) -> Tuple[str, float]:
    with torch.no_grad():
        outputs = model(input_tensor)
    outputs = outputs.cpu() 
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    top_p, top_class_index = probabilities.topk(1, dim=0)
    return class_names[top_class_index.item()], top_p.item() * 100

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    try:
        PLANT_CLASSES = load_class_names(CLASS_TO_ID_JSON, SPECIES_NAME_JSON, NUM_CLASSES)
        model = load_plant_model(NUM_CLASSES, WEIGHTS_PATH)
        input_tensor = preprocess_image(IMAGE_PATH)
        
        start_time = time.time()
        predicted_class, confidence = classify_plant(model, input_tensor, PLANT_CLASSES)
        end_time = time.time()
        
        print("\n==============================")
        print("✨ 植物识别结果 ✨")
        print(f"运行设备: {device}")
        print(f"输入图片: {IMAGE_PATH}")
        print(f"预测类别: **{predicted_class}**")
        print(f"置信度: **{confidence:.2f}%**")
        print(f"耗时: {(end_time - start_time):.4f} 秒")
        print("==============================")

    except Exception as e:
        print(f"\n[程序终止] 错误信息: {e}") 