from urllib.parse import urljoin

import re
import requests

from bs4 import BeautifulSoup


def main():
    headers = {'user-agent': 'Baiduspider'}
    proxies = {
        'http': 'http://122.114.31.177:808'
    }
    base_url = 'https://www.zhihu.com/'
    seed_url = urljoin(base_url, 'explore')   #可以改为hot
    resp = requests.get(seed_url,             #获得网页源码
                        headers=headers,
                        proxies=proxies)
    soup = BeautifulSoup(resp.text, 'lxml')   #源码文本方式存储
    href_regex = re.compile(r'^/question')    #寻找所有question标签的元素
    print(href_regex)
    '''初始化空字典'''
    link_set = set()
    for a_tag in soup.find_all('a', {'href': href_regex}):#拼接成链接
        if 'href' in a_tag.attrs:
            href = a_tag.attrs['href']
            full_url = urljoin(base_url, href)
            link_set.add(full_url)
    print('Total %d question pages found.' % len(link_set))
    #print(link_set)

if __name__ == '__main__':
    main()