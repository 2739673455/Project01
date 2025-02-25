import requests


def mymemory_translate(text, langpair):
    # 使用MyMemory免费API
    url = "https://api.mymemory.translated.net/get"
    # 参数传入待翻译的文本，翻译对
    params = {"q": text, "langpair": langpair}
    # 发送请求
    resp = requests.get(url=url, params=params)
    # 解析响应
    return resp.json()["responseData"]["translatedText"]


text = "刚出锅的馒头一块钱一个！"
# 汉语->英语
text_en = mymemory_translate(text, "zh|en")
# 英语->意大利语
text_it = mymemory_translate(text_en, "en|it")
# 意大利语->汉语
text_zh = mymemory_translate(text_it, "it|zh")
print(text_zh)
