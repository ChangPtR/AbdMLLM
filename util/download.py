import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_gif(url, save_dir):
    try:
        filename = url.split('/')[-1]
        save_path = os.path.join(save_dir, filename)

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return url, True  # 已存在则跳过
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        print(f"下载: {url}")
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)

        return url, True
    except Exception as e:
        print(e)
        return url, False

def load_urls(tsv_path):
    urls = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts and parts[0].endswith('.gif'):
                urls.append(parts[0])
    return urls

def main(tsv_path, save_dir, max_workers=64):
    os.makedirs(save_dir, exist_ok=True)
    urls = load_urls(tsv_path)

    failed_urls = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_gif, url, save_dir): url for url in urls}

        for future in tqdm(as_completed(futures), total=len(futures)):
            url, success = future.result()
            if not success:
                failed_urls.append(url)

    if failed_urls:
        print(f"\n共有 {len(failed_urls)} 个下载失败，保存至 failed_urls.txt")
        with open('failed_urls.txt', 'w', encoding='utf-8') as f:
            for url in failed_urls:
                f.write(url + '\n')

if __name__ == "__main__":
    main(tsv_path="dVAR/TGIF-Release-master/data/tgif-v1.0.tsv", save_dir="/newdata1/xw/tgif")
