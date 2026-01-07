import requests

def wikidata_api():
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": "Obama"
    }
    headers = {
        "User-Agent": "TestScript/1.0 (me@example.com) Python/3.9"
    }
    proxies = {
        "http": "http://127.0.0.1:1080",
        "https": "http://127.0.0.1:1080",
    }
    try:
        response = requests.get(url, params=params, headers=headers, proxies=proxies, timeout=30)
        response.raise_for_status()
        print("[SUCCESS] API 正常，返回内容如下：")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"[FAILURE] API 请求失败: {e}")

wikidata_api()
