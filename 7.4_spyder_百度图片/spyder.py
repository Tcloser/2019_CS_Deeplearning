# -*- coding:utf-8 -*-
import re
import requests


def dowmloadPic(html, keyword,i):
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)
    #pic_url = re.findall('src=\".*?(.*?jpeg|.*?png)\"', html, re.S)
    #pic_url = re.findall("h(?<=[http])[^()]+[^()]+(?=[\.])+\.png|.jpeg|.jpg|.gif",html)
    #pic_url = re.findall('data-imgurl=:',html)
    #pic_ = re.findall("(jpg|jpeg|png|gif",pic_url)
    #pic_ual = 'u' + pic_ual
    #i = 1
    print('找到关键词:' + keyword + '的图片，现在开始下载图片...' + str(len(pic_url)))  # +"(OO)" + pic_)
    for each in pic_url:
        print('正在下载第' + str(i) + '张图片，图片地址:' + str(each))
        try:
            pic = requests.get(each, timeout=3,allow_redirects=False)
        except requests.exceptions.ConnectionError:
            print('【错误】当前图片无法下载')
            continue
        except requests.exceptions.ReadTimeout:
            print('超时')
            continue

        dir = 'image/' + keyword + '_' + str(i) + '.jpg'
        fp = open(dir, 'wb')
        fp.write(pic.content)
        fp.close()
        i += 1
    return i


if __name__ == '__main__':
    word = "生气烦恼"
    pages = 10
    #url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&ct=201326592&v=flip'
    i = 0
    for page in range(pages):
        #url = 'http://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word='+word
        url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn=' + str(page*70) + '&gsm=3c&ct=&ic=0&lm=-1&width=0&height=0'
        #word = "test"
        result = requests.get(url,allow_redirects=False)
        i = dowmloadPic(result.text, word,i)
