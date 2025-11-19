import json
import os
import time
import random
import requests
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import urllib3
from deep_translator import GoogleTranslator

# 禁用安全警告，干就完了
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 配置 =================
SOURCE_JSON = 'plantnet300K_species_id_2_name.json'
TARGET_JSON = 'plant_dictionary_zh.json'

# 强力伪装头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://baike.baidu.com/',
    'Accept-Language': 'zh-CN,zh;q=0.9'
}

# 翻译器备用
translator = GoogleTranslator(source='auto', target='zh-CN')

def safe_translate(text):
    """谷歌翻译兜底"""
    try:
        # 只翻译拉丁名部分，去掉作者
        clean_text = text.split('(')[0].strip()
        return translator.translate(clean_text)
    except:
        return text

def fetch_baidu_page(url):
    """通用请求函数"""
    try:
        res = requests.get(url, headers=HEADERS, timeout=10, verify=False, allow_redirects=True)
        res.encoding = 'utf-8'
        return res
    except:
        return None

def parse_item_page(html):
    """解析具体的词条页面"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # 1. 抓标题 (中文名)
    zh_name = None
    h1 = soup.find('h1')
    if h1:
        zh_name = h1.get_text().strip()
    
    # 如果标题是拉丁文，尝试在 Infobox 里找 "中文名"
    basic_infos = soup.find_all('dt', class_='basicInfo-item')
    for dt in basic_infos:
        if "中文名" in dt.get_text():
            next_dd = dt.find_next_sibling('dd')
            if next_dd:
                zh_name = next_dd.get_text().strip()
                break
    
    # 2. 抓描述 (摘要)
    desc = "暂无详细描述。"
    summary = soup.find('div', class_='lemma-summary') or soup.find('div', class_='J-summary')
    if summary:
        desc = summary.get_text().strip().replace("\n", "")
        desc = re.sub(r'\[.*?\]', '', desc) # 去掉 [1][2]
    
    return zh_name, desc

def get_plant_info(latin_name):
    clean_name = " ".join(latin_name.split()[:2]).strip()
    
    # --- 第一步：发起搜索 ---
    search_url = f"https://baike.baidu.com/search/word?word={clean_name}"
    response = fetch_baidu_page(search_url)
    
    if not response:
        return None

    final_html = response.text
    final_url = response.url

    # --- 第二步：判断是否需要“二级跳转” ---
    # 如果 URL 包含 /search/，说明没有直接跳到词条，而是展示了搜索结果列表
    if "/search/" in final_url:
        soup = BeautifulSoup(final_html, 'html.parser')
        
        # 寻找搜索结果的第一条链接
        # 百度搜索结果通常在 a.result-title 或 h3 > a
        # 这里尝试抓取第一个结果
        first_result = soup.find('a', class_='result-title')
        
        if first_result and first_result.get('href'):
            # 找到了！比如 "毒莴苣" 的链接
            target_link = first_result['href']
            if not target_link.startswith('http'):
                target_link = "https://baike.baidu.com" + target_link
            
            # 再次请求这个具体的词条页
            # print(f" -> 追踪跳转: {target_link}")
            sub_res = fetch_baidu_page(target_link)
            if sub_res:
                final_html = sub_res.text
            else:
                return None
        else:
            # 搜索结果页都没有东西，那就是真没有了
            return None

    # --- 第三步：解析页面 ---
    zh_name, desc = parse_item_page(final_html)
    
    # 校验：如果名字没取到，或者描述是空的
    if not zh_name: 
        return None
        
    return {
        "zh_name": zh_name,
        "desc": desc,
        "habit": "详情请见描述。"
    }

def run_crawler():
    if not os.path.exists(SOURCE_JSON):
        print("❌ 找不到源文件")
        return

    with open(SOURCE_JSON, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
        
    # 读取现有进度
    final_dict = {}
    if os.path.exists(TARGET_JSON):
        with open(TARGET_JSON, 'r', encoding='utf-8') as f:
            try: 
                final_dict = json.load(f) 
            except: 
                pass

    print(f"📂 当前库中有 {len(final_dict)} 条。")

    # --- 清洗无效数据 ---
    # 这次我们狠一点，只要名字包含拉丁文(即没翻译成功)或者描述含糊的，全部重抓
    keys_to_fix = []
    for k, v in final_dict.items():
        # 规则：如果中文名含有 Latin (比如 'Lactuca') 且描述是 '本地数据库暂未...'，则视为失败，删掉重来
        if "本地数据库暂未" in v['desc']:
            keys_to_fix.append(k)
    
    for k in keys_to_fix:
        del final_dict[k]
        
    if keys_to_fix:
        print(f"♻️  自动删除了 {len(keys_to_fix)} 条之前的垃圾数据，准备重新获取...")

    # 任务列表
    all_latin_names = list(source_data.values())
    todo_list = [name for name in all_latin_names if " ".join(name.split()[:2]).strip() not in final_dict]
    
    print(f"🚀 开始暴力抓取 {len(todo_list)} 个词条...")
    
    counter = 0
    for latin_name in tqdm(todo_list):
        clean_name = " ".join(latin_name.split()[:2]).strip()
        
        # 1. 爬百度
        info = get_plant_info(latin_name)
        
        if info:
            final_dict[clean_name] = info
        else:
            # 2. 百度彻底失败 -> 启用谷歌翻译
            # print(f" -> 百度无结果，调用翻译: {clean_name}")
            trans_name = safe_translate(clean_name)
            final_dict[clean_name] = {
                "zh_name": trans_name, # 至少这里是中文！
                "desc": "暂无详细百科资料（已自动翻译名称）。",
                "habit": "未知"
            }
        
        counter += 1
        if counter % 5 == 0:
            with open(TARGET_JSON, 'w', encoding='utf-8') as f:
                json.dump(final_dict, f, ensure_ascii=False, indent=4)
        
        time.sleep(random.uniform(0.5, 1.2))

    with open(TARGET_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=4)
    print("\n✅ 所有数据抓取/修复完成！")

if __name__ == "__main__":
    run_crawler()