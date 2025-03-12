from langchain_community.utils.user_agent import get_user_agent
from tqdm import tqdm
from bs4 import BeautifulSoup
import hashlib
from itertools import accumulate
import requests
import json
from embeddings import (
    PineconeStore,
    embedding_doc,
)
from huggingface_hub.utils._http import HfHubHTTPError
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
import os
import time

default_header_template = {
    "User-Agent": get_user_agent(),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
    ";q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
international_links = [
    'https://oiss.skku.edu/oiss/life/visa.do',
    'https://oiss.skku.edu/oiss/life/insurance.do',
    'https://oiss.skku.edu/oiss/life/parttimeworks.do',
    'https://oiss.skku.edu/oiss/life/immigration.do',
    'https://oiss.skku.edu/oiss/life/campus.do',
    'https://oiss.skku.edu/oiss/life/administrative.do',
    'https://oiss.skku.edu/oiss/life/accommodation.do',
    'https://oiss.skku.edu/oiss/life/library.do',
    'https://oiss.skku.edu/oiss/life/facilities.do',
    'https://oiss.skku.edu/oiss/life/scholarship.do',
    'https://oiss.skku.edu/oiss/life/certificate.do',
    'https://oiss.skku.edu/oiss/life/employment.do',
    'https://oiss.skku.edu/oiss/life/others.do',
    'https://oiss.skku.edu/oiss/academics/course_registration.do',
    'https://oiss.skku.edu/oiss/academics/registration.do',
    'https://oiss.skku.edu/oiss/academics/reinstatement.do',
    'https://oiss.skku.edu/oiss/academics/dismissal.do',
    'https://oiss.skku.edu/oiss/academics/under_curriculum.do',
    'https://oiss.skku.edu/oiss/academics/liberal_curriculum.do',
    'https://oiss.skku.edu/oiss/academics/major_curriculum.do',
    'https://oiss.skku.edu/oiss/academics/grad_curriculum.do',
    'https://oiss.skku.edu/oiss/academics/course.do',
    'https://oiss.skku.edu/oiss/academics/grading.do',
    'https://oiss.skku.edu/oiss/academics/under_plural.do',
    'https://oiss.skku.edu/oiss/academics/combined.do',
    'https://oiss.skku.edu/oiss/academics/graduation.do',
    'https://oiss.skku.edu/oiss/academics/thesis.do',
]
skku_links = [
    'https://www.skku.edu/skku/campus/support/welfare_11_1.do?mode=info&conspaceCd=20201040&srResId=11&srShowTime=W&srCategory=L',
    'https://www.skku.edu/skku/campus/support/welfare_11_1.do?mode=info&conspaceCd=20201251&srResId=12&srShowTime=D&srCategory=L',
    'https://www.skku.edu/skku/campus/support/welfare_11_1.do?mode=info&conspaceCd=20201104&srResId=3&srShowTime=W&srCategory=L',
    'https://www.skku.edu/skku/campus/support/welfare_11.do?mode=info&conspaceCd=10201030&srResId=1&srShowTime=W&srCategory=L',
    "https://www.skku.edu/skku/campus/support/welfare_11.do?mode=info&conspaceCd=10201031&srResId=2&srShowTime=D&srCategory=L",
    "https://www.skku.edu/skku/campus/support/welfare_11.do?mode=info&conspaceCd=10201034&srResId=4&srShowTime=W&srCategory=L",
    "https://www.skku.edu/skku/campus/support/welfare_11.do?mode=info&conspaceCd=10201032&srResId=5&srShowTime=D&srCategory=L",
    "https://www.skku.edu/skku/campus/support/welfare_11.do?mode=info&conspaceCd=10201033&srResId=6&srShowTime=D&srCategory=L",
]
headers = ['h1', 'h2', 'h3', 'h4', 'h5']


def get_header(cur_col, headers):

    colspans = [int(h['colspan']) for h in headers] 
    cumulative_sum = list(accumulate(colspans))
    # assert cur_col < cumulative_sum[-1], f'invalid cur_col: {cur_col}'
    for i, sum in enumerate(cumulative_sum):
        if cur_col < sum:
            return headers[i]
    return headers[0] 


def get_table(content):
    headers = []
    thead = content.find_all('thead')[0]
    for th in thead.find_all('th'):
        header = th.get_text(strip=True)
        rowspan = int(th.get('rowspan', 1))
        colspan = int(th.get('colspan', 1))
        headers.append(
            {
                'header': header,
                'rowspan': rowspan,
                'colspan': colspan
            } 
        )
    # print(headers)
    if not headers:
        return ''

    tbody = content.find('tbody')
    
    elements = [{} for i in range(len(tbody.find_all('tr')))]

    for i, tr in enumerate(tbody.find_all('tr')):
        cur_col = 0
        for j, td in enumerate(tr.find_all('td')):
            val = td.text.strip().replace('\t', '')
            rowspan = int(td.get('rowspan', 1))
            colspan = int(td.get('colspan', 1))

            
            while True:
                cur_header = get_header(cur_col, headers)

                if elements[i].get(cur_header['header'], None) is None:
                    for row in range(rowspan):
                        elements[i+row][cur_header['header']] = [val, colspan]
                    
                    if colspan == cur_header['colspan']:
                        cur_col += colspan
                    break
                else:
                    cur_col += elements[i][cur_header['header']][1]
                    if colspan != cur_header['colspan']:
                        elements[i][cur_header['header']][0] += ': '+val
                        cur_col += colspan
                        break
            
    elements = [
        {
            k: v[0] 
            for k, v in ele.items()
        }
        for ele in elements
    ]
    return json.dumps(elements, indent=4)

def is_modified(hashed_pages, content):
    current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    if current_hash in hashed_pages:
        return False
    else:
        return True

def update_hashed_pages(page_content, url):
    current_hash = hashlib.md5(page_content.encode('utf-8')).hexdigest()
     
    hashed_pages = []
    with open('./hashed_pages.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['url'] == url:
                continue
            hashed_pages.append(data)
    hashed_pages.append({'hash':current_hash, 'url':url})

    with open('./hashed_pages.jsonl', 'w') as f:
        for data in hashed_pages:
            json.dump(data, f)
            f.write('\n')
    
    

def scrap(urls):
    session = requests.Session()
    header_template = default_header_template.copy()
    session.headers = dict(header_template)
    session.verify = True

    modified_urls = []
    hashed_pages = []
    with open('./hashed_pages.jsonl', 'r') as f:
        for line in f:
            hashed_pages.append(json.loads(line)['hash'])

    results = []
    with open('./results.jsonl', 'r', encoding='utf-8') as f1: 
        for line in f1:
            results.append(json.loads(line))

    for url in tqdm(urls):
        response =session.get(url)

        # check if the web page is modified
        page_content = response.text
        if not is_modified(hashed_pages, page_content):
            continue


        modified_urls.append(url)
        update_hashed_pages(page_content, url)
        results = [r for r in results if r['link'] != url]
        soup = BeautifulSoup(response.content, 'html.parser')

        # title = soup.title.text.replace('/', '')
        title = soup.find(class_='page-title').get_text().strip().replace('/', '')

        contents = soup.find(class_='content')
        if contents.find(class_='content-box'):
            contents = contents.find(class_='content-box')

        contents = contents.find_all(True, recursive=False)

        i = 0
        chunk = ''
        header1 = ''
        header2 = ''
        texts = ''

        for content in contents:
            # headers
            if content.name == 'h4':
                if i == 0:
                    header1 = content.get_text().strip().replace('\t', '').replace('\n', '')
                    i += 1
                    continue
                with open(f'./results/{title}-{i}.txt', 'w', encoding='utf-8') as f:
                    chunk = f"{title}\n{header1}\n{header2}\n\n{texts}"
                    f.write(chunk)
                    json_chunk = {
                        'title': f'{title}-{i}',
                        'content': chunk,
                        'link': url
                    }
                    results.append(json_chunk)
                    i += 1
                    texts = ''
                header1 = content.get_text().strip().replace('\t', '').replace('\n', '')
                continue

            elif content.name == 'h5':
                with open(f'./results/{title}-{i}.txt', 'w', encoding='utf-8') as f:
                    chunk = f"{title}\n{header1}\n{header2}\n\n{texts}"
                    f.write(chunk)
                    json_chunk = {
                        'title': f'{title}-{i}',
                        'content': chunk,
                        'link': url
                    }
                    results.append(json_chunk)
                    i += 1
                    texts = ''
                header2 = content.get_text().strip().replace('\t', '').replace('\n', '')
                continue

            # content
            if content.name == 'table':
                text = get_table(content)

            elif 'scrollbox' in content.get('class', []):
                content = content.find('table')
                text = get_table(content)

            else:
                text = content.get_text()
                text = text.strip().replace('\t', '').replace('\n\n', '\n')

            texts += text

        with open(f'./results/{title}-{i}.txt', 'w', encoding='utf-8') as f:
            chunk = f"{title}\n{header1}\n{header2}\n\n{texts}"
            f.write(chunk)
            json_chunk = {
                'title': f'{title}-{i}',
                'content': chunk,
                'link': url
            }
            results.append(json_chunk)
    

        with open('./results.jsonl', 'w', encoding='utf-8') as f: 
            for result in results:
                json.dump(result, f)
                f.write('\n')
     
    return modified_urls


def scrap2(urls):
    session = requests.Session()
    header_template = default_header_template.copy()
    session.headers = dict(header_template)
    session.verify = True

    contents = []
    modified_urls = [urls[0]]
    for url in modified_urls:
        response =session.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        cafeteria = soup.find(class_='info_tit').text.strip()
        menu = soup.find_all('pre')[1:]

        day = ['월', '화', '수', '목', '금']
        menu_list = []
        for d, m in zip(day, menu):
            menu_list.append('\n'+d+'\n'+m.text+'\n')

        chunk = '식당이름: ' + cafeteria + '\n메뉴\n' + ''.join(menu_list)
        json_chunk = {
            'title': f'{cafeteria}',
            'content': chunk,
            'link': url
        }
        contents.append(json_chunk)

    with open('./results.jsonl', 'a') as f:
        for content in contents:
            json.dump(content, f)
            f.write('\n') 

    return modified_urls



def dbupdate():
    # scrap modified web pages
    modified_urls1 = scrap(international_links)
    modified_urls2 = scrap2(skku_links)
    modified_urls = modified_urls1 + modified_urls2
    print(modified_urls)


    # get modified chunks
    modified_chunks = []
    with open('./results.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            if chunk['link'] in modified_urls:
                modified_chunks.append(chunk)
        
    # create embeddings
    hf_embeddings = HuggingFaceEndpointEmbeddings(
        model="intfloat/multilingual-e5-large-instruct",
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
    )
    vector_store = PineconeStore('iskku-data3')
    for chunk in tqdm(modified_chunks):
        attempts = 1
        while True:
            try:
                embeddings, hf_embeddings = embedding_doc(chunk['content'], model_name = "intfloat/multilingual-e5-large-instruct", hf_embeddings=hf_embeddings)
                vector_store.save_vectors(embeddings, chunk['title'], chunk['content'], chunk['link'])
                break
            except HfHubHTTPError as e:
                print(f'Attempts: {attempts}')
                if attempts > 3:
                    print(f"Too large: {chunk['title']}")
                    break
                attempts += 1
                time.sleep(60)
            except:
                pass