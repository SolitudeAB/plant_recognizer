import json
import os

SPECIES_NAME_JSON = 'plantnet300K_species_id_2_name.json'
CLASS_TO_ID_JSON = 'class_idx_to_species_id.json' 

def test_json_load(file_path):
    print(f"尝试加载文件: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件 {file_path} 不存在。")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ 成功加载 {file_path}。数据类型: {type(data)}。包含 {len(data)} 个键/元素。")
        
    except FileNotFoundError:
        print(f"❌ 错误: 运行时仍然找不到文件 {file_path}。检查路径！")
    except json.JSONDecodeError as e:
        print(f"❌ 错误: {file_path} 不是有效的 JSON 文件！请重新下载。错误信息: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

test_json_load(CLASS_TO_ID_JSON)
test_json_load(SPECIES_NAME_JSON)