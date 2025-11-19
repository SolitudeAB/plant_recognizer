import json
import os
import time
from deep_translator import GoogleTranslator
from tqdm import tqdm

# ================= 配置 =================
SOURCE_JSON = 'plantnet300K_species_id_2_name.json'
TARGET_JSON = 'plant_dictionary_zh.json'

# ================= 1. 核心精选库 (手动录入的高质量数据) =================
# 这些数据会直接写入，不做翻译，保证演示时的关键植物信息准确
MANUAL_DATA = {
    "Lactuca virosa": {
        "zh_name": "毒莴苣 (野莴苣)",
        "desc": "毒莴苣（学名：Lactuca virosa）是菊科莴苣属的一种二年生草本植物。它含有一种乳白色的汁液，具有类似鸦片的镇静和轻微麻醉作用，历史上曾被用作止痛药。原产于欧洲，现已广泛分布。",
        "habit": "通常生长在河岸、路边或受干扰的土壤中。茎直立，可能有刺，花朵为淡黄色。"
    },
    "Cirsium vulgare": {
        "zh_name": "翼蓟 (欧洲蓟)",
        "desc": "翼蓟（学名：Cirsium vulgare）是菊科蓟属的植物。它是常见的野外杂草，具有极强的生命力。叶片边缘有尖锐的刺，花头呈紫色，是蜜蜂和蝴蝶喜爱的蜜源植物。",
        "habit": "喜阳光充足的环境，耐旱，多见于牧场、荒地和路边。"
    },
    "Rosa chinensis": {
        "zh_name": "月季花",
        "desc": "月季花（学名：Rosa chinensis），被称为花中皇后，又称“月月红”，是常绿、半常绿低矮灌木，四季开花。原产中国，是现代玫瑰育种的重要亲本。",
        "habit": "适应性强，耐寒耐旱，喜富含有机质、排水良好的微酸性土壤。"
    },
    "Helianthus annuus": {
        "zh_name": "向日葵",
        "desc": "向日葵（学名：Helianthus annuus）是菊科向日葵属的植物。因花序随太阳转动而得名。原产北美洲，世界各地均有栽培，种子可榨油。",
        "habit": "一年生草本，高1～3.5米。喜温又耐寒，对土壤要求不严。"
    },
    "Taraxacum officinale": {
        "zh_name": "西洋蒲公英",
        "desc": "西洋蒲公英（学名：Taraxacum officinale）是菊科蒲公英属植物，广泛分布于北半球温带地区。它是最常见的野草之一，叶片呈锯齿状。",
        "habit": "多年生草本，花黄色。果实成熟后像白色绒球，随风飘散。"
    },
     "Acer platanoides": {
        "zh_name": "挪威枫",
        "desc": "挪威枫（学名：Acer platanoides）是无患子科槭属的落叶乔木。原产于欧洲东部和中部。叶片宽大，秋季变为亮黄色或橙红色，是优秀的行道树。",
        "habit": "耐阴性较强，耐寒，生长迅速，但具有一定的入侵性。"
    }
}

# ================= 2. 工具函数 =================

def translate_latin(latin_name):
    """调用谷歌翻译将拉丁名转中文"""
    try:
        # 清理：去掉作者名，只保留前两个词 (Genus species)
        clean_latin = " ".join(latin_name.split()[:2]).strip()
        # 翻译
        translator = GoogleTranslator(source='auto', target='zh-CN')
        zh_name = translator.translate(clean_latin)
        
        # 简单的校验：如果翻译结果全是英文，说明翻译失败，返回原名
        if all(ord(c) < 128 for c in zh_name):
            return clean_latin
        return zh_name
    except:
        # 如果网络报错，返回拉丁名
        return " ".join(latin_name.split()[:2]).strip()

def generate_template_desc(zh_name, latin_name):
    """生成标准化的描述模板"""
    clean_latin = " ".join(latin_name.split()[:2]).strip()
    return f"{zh_name}（学名：{clean_latin}）是记录于 Pl@ntNet 数据库中的一种植物。该数据由 AI 自动翻译生成，详细生物学特征请参考专业图鉴。"

# ================= 3. 主逻辑 =================

def build_dictionary():
    if not os.path.exists(SOURCE_JSON):
        print("❌ 错误：找不到源文件 plantnet300K_species_id_2_name.json")
        return

    print("📂 读取源文件...")
    with open(SOURCE_JSON, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # 提取所有拉丁名
    all_latin_names = list(source_data.values())
    total = len(all_latin_names)
    
    final_dict = {}
    
    print(f"🚀 开始构建字典，共 {total} 个条目...")
    print("☕ 这可能需要几分钟，请耐心等待...")

    for latin_name in tqdm(all_latin_names):
        # 1. 清洗键名 (作为字典的 Key)
        key_name = " ".join(latin_name.split()[:2]).strip()
        
        # 2. 优先检查【精选库】
        if key_name in MANUAL_DATA:
            final_dict[key_name] = MANUAL_DATA[key_name]
            continue
            
        # 3. 如果不在精选库，进行【自动构建】
        # 为了速度和稳定性，我们先尝试翻译
        zh_name = translate_latin(key_name)
        
        # 构建条目
        entry = {
            "zh_name": zh_name,
            "desc": generate_template_desc(zh_name, latin_name),
            "habit": "未知（自动生成数据）"
        }
        
        final_dict[key_name] = entry
        
        # 极短的延时，避免 API 限制
        time.sleep(0.1)

    # ================= 4. 保存 =================
    print(f"\n💾 正在保存到 {TARGET_JSON} ...")
    with open(TARGET_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=4)
        
    print("✅ 成功！字典已生成完毕。现在您可以离线运行 app.py 了。")

if __name__ == "__main__":
    build_dictionary()