#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# libraries for basic
from flask import Flask, render_template, request
from data import YouTubeData

# libraries for crawling_navershopping_name,price,img
from bs4 import BeautifulSoup
import time

from selenium import webdriver

# libraries for crawling_navershopping_cheapest price
from selenium import webdriver
from bs4 import BeautifulSoup
from html_table_parser import parser_functions as parser
from time import sleep
import requests
import pandas as pd

# libraries for sentiment analysis
import re
import pandas as pd
import numpy as np
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model

# libraries for nlp(recommendation)

from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
import itertools
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# start
DEVELOPMENT_ENV  = True

app = Flask(__name__)

app_data = {
    "name":         "Jeong's Flask Web App",
    "description":  "This Page is for",
    "author":       "Giyoung Jeong",
    "html_title":   "Jeong's BigData Final Project on Flask",
    "project_name": "S.YOL.C",
    "keywords":     "flask, webapp, template, basic"
}


@app.route('/')
def index():
    return render_template('index.html', app_data=app_data)


@app.route('/about')
def about():
    return render_template('about.html', app_data=app_data)


@app.route('/service1')
def service1():
    return render_template('service1.html', app_data=app_data)

@app.route('/service2')
def service2():
    return render_template('service2.html', app_data=app_data)

@app.route('/search2', methods=['POST'])
def search2():
    if request.method == 'POST':
        # getting the video details like images and channelid
        search = request.form['search']
        data = YouTubeData(search)
        snippet = data.get_channel_details(search)
        return render_template('search2_page.html', message=snippet, search=search, app_data=app_data)
    else:
        return render_template('service2.html', app_data=app_data)

@app.route('/get_more2/<channelId>/<search>/<videoid>', methods=['GET', 'POST'])
def get_more2(channelId, search, videoid):
    ### for result of youtbue search
    if request.method == 'GET':
        data = YouTubeData(search)
        content = data.get_channel_stats(channelId)
        snippet = data.get_videoDetails(videoid)
        stats = data.get_statistics(videoid)

        ##### result of navershopping
        shop_search = '보다나 물결 고데기 블루 40'

        search_list = []
        search_list_src = []
        search_list_price = []

        '''
        # 드라이버 옵션 생성
        options = webdriver.ChromeOptions()
        # 창 숨기는 옵션 추가
        options.add_argument("headless")
        '''

        driver = webdriver.Chrome("C:/Users/admin/Desktop/chromedriver_win32/chromedriver.exe"
                                  #, options=options
                                  )

        # 3초 기다려주기, 웹페이지 로딩까지
        driver.implicitly_wait(3)

        # 드라이버 실행
        driver.get("https://search.shopping.naver.com/search/all?query=" + shop_search)

        y = 2
        for timer in range(0, 5):
            driver.execute_script("window.scrollTo(0, " + str(y) + ")")
            y = y + 1000
            time.sleep(1)

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        select = "#__next > div > div.style_container__1YjHN > div.style_inner__18zZX > div.style_content_wrap__1PzEo > div.style_content__2T20F > ul > div"

        # text
        for i in soup.select(select)[0].find_all("div", class_='basicList_title__3P9Q7'):
            # print(i.text)
            search_list.append(i.text)

        # img
        for i in soup.select(select)[0].find_all("div", class_='thumbnail_thumb_wrap__1pEkS _wrapper'):
            tmp = i.find_all("img")
            for index in tmp:
                search_list_src.append(index['src'])

        # price
        for i in soup.select(select)[0].find_all("span", class_='price_num__2WUXn'):
            #print(i.text)
            search_list_price.append(i.text)

        driver.close()

        #########################################################################################
        ####################crawling_navershopping_cheapest price################################
        #########################################################################################

        name = ['보다나 트리플 플로우 물결 고데기 크리미블루 40형']
        category = ['별점']

        ns_address = "https://search.shopping.naver.com/catalog/28640667554?query=%EB%B3%B4%EB%8B%A4%EB%82%98%20%ED%8A%B8%EB%A6%AC%ED%94%8C%20%ED%94%8C%EB%A1%9C%EC%9A%B0%20%EB%AC%BC%EA%B2%B0%20%EA%B3%A0%EB%8D%B0%EA%B8%B0%20%ED%81%AC%EB%A6%AC%EB%AF%B8%EB%B8%94%EB%A3%A8%2040%ED%98%95&NaPm=ct%3Dl4i2oz3s%7Cci%3D2f3c84ac58c9161866b5a0ef7228d739e6b72414%7Ctr%3Dslsl%7Csn%3D95694%7Chk%3D9bd6e369589d5ca6dc17f187c8b39f2c57a36cb9"

        # xpath
        shoppingmall_review = "/html/body/div/div/div[2]/div[2]/div[2]/div[3]/div[6]/ul"

        header = {'User-Agent': ''}
        d = webdriver.Chrome('C:/Users/admin/Desktop/chromedriver_win32/chromedriver.exe')  # webdriver = chrome
        d.implicitly_wait(3)
        d.get(ns_address)
        req = requests.get(ns_address, verify=False)
        html = req.text
        soup = BeautifulSoup(html, "html.parser")
        sleep(2)

        d.close()

        select_table = '#__next > div > div.style_container__3iYev > div.style_inner__1Eo2z > div.style_content_wrap__2VTVx > div.style_content__36DCX > div > div.summary_info_area__3XT5U > div.condition_area > table'

        temp = soup.select(select_table)

        p = parser.make2d(temp[0])

        df = pd.DataFrame(p[1:], columns=p[0])
        result_cheapest = df.iloc[0:1, :2]

        result_cheapest = result_cheapest.to_html()


        ##########################################################################################
        ##########################################################################################
        ##########################################################################################
        ### for result of sentiment analysis

        mecab = Mecab('C:/mecab/mecab-ko-dic')

        total_data = pd.read_table('C:/Sources/Projects/빅데이터과정_파이널project_유튜브 화장품 정보/res/300data_straightener/ratings_total.txt', names=['ratings', 'reviews'])
        total_data['label'] = np.select([total_data.ratings > 3], [1], default=0)

        # reviews 열에서 중복인 내용이 있다면 중복 제거
        total_data.drop_duplicates(subset=['reviews'], inplace=True)
        train_data, test_data = train_test_split(total_data, test_size=0.25, random_state=42)

        # 한글과 공백을 제외하고 모두 제거
        train_data['reviews'] = train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
        train_data['reviews'].replace('', np.nan, inplace=True)
        test_data.drop_duplicates(subset=['reviews'], inplace=True)
        test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행

        test_data['reviews'].replace('', np.nan, inplace=True)
        test_data = test_data.dropna(how='any')
        stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯',
                     '지', '임', '게']

        train_data['tokenized'] = train_data['reviews'].apply(mecab.morphs)
        train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
        test_data['tokenized'] = test_data['reviews'].apply(mecab.morphs)
        test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

        negative_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values)
        positive_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values)
        negative_word_count = Counter(negative_words)
        positive_word_count = Counter(positive_words)

        X_train = train_data['tokenized'].values
        y_train = train_data['label'].values
        X_test = test_data['tokenized'].values
        y_test = test_data['label'].values

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)

        threshold = 2
        total_cnt = len(tokenizer.word_index)  # 단어의 수
        rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
        total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
        rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
        # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
        for key, value in tokenizer.word_counts.items():
            total_freq = total_freq + value

            # 단어의 등장 빈도수가 threshold보다 작으면
            if (value < threshold):
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value
        # 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
        # 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
        vocab_size = total_cnt - rare_cnt + 2
        tokenizer = Tokenizer(vocab_size, oov_token='OOV')
        tokenizer.fit_on_texts(X_train)
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        def below_threshold_len(max_len, nested_list):
            count = 0
            for sentence in nested_list:
                if (len(sentence) <= max_len):
                    count = count + 1
            print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (count / len(nested_list)) * 100))

        max_len = 80

        X_train = pad_sequences(X_train, maxlen=max_len)
        X_test = pad_sequences(X_test, maxlen=max_len)

        path = "C:/Sources/Projects/빅데이터과정_파이널project_유튜브 화장품 정보/res/300data_straightener/result/sensitivityAnalysis/sensitivityAnalysis_best_model_forNavershopping_Review.h5"
        loaded_model = load_model(path)

        def sentiment_predict(new_sentence):
            new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
            new_sentence = mecab.morphs(new_sentence)
            new_sentence = [word for word in new_sentence if not word in stopwords]
            encoded = tokenizer.texts_to_sequences([new_sentence])
            pad_new = pad_sequences(encoded, maxlen=max_len)

            score = float(loaded_model.predict(pad_new, verbose=0))
            if (score > 0.5):
                print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
            else:
                print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))

        tmp = pd.read_csv(
            'C:/Sources/Projects/빅데이터과정_파이널project_유튜브 화장품 정보/res/300data_straightener/result/sensitivityAnalysis/crawledData_naverReviews_VodanaStraightner_blue40.csv')

        def sentiment_predict_NaverShoppingReviews(new_sentence):
            new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
            new_sentence = mecab.morphs(new_sentence)
            new_sentence = [word for word in new_sentence if not word in stopwords]
            encoded = tokenizer.texts_to_sequences([new_sentence])
            pad_new = pad_sequences(encoded, maxlen=max_len)

            score = float(loaded_model.predict(pad_new, verbose=0))

            if (score > 0.5):
                positive_reviews.append(score)
            else:
                negative_reviews.append(score)

            return positive_reviews, negative_reviews

        reviews = tmp['review']

        positive_reviews = list()
        negative_reviews = list()

        def rate_n_or_p_reviews(reviews):
            for review in reviews:
                positive_reviews, negative_reviews = sentiment_predict_NaverShoppingReviews(review)
            total = len(positive_reviews) + len(negative_reviews)
            positive_result = f"positive : {len(positive_reviews) / total * 100} %"
            negative_result = f"negative : {len(negative_reviews) / total * 100} %"
            return positive_result , negative_result

        positive_result, negative_result  = rate_n_or_p_reviews(reviews)

        ##########################################################################################
        ##########################################################################################
        ##########################################################################################
        ###### keyword(hashtag) part #####

        srt = YouTubeTranscriptApi.get_transcript(videoid, languages=['ko'])

        ###
        text = ''

        for i in range(len(srt)):
            text += srt[i]['text'] + ' '  # text 부분만 가져옴

        text_ = text.replace(' ', '')

        ###
        texts = list()

        for i in range(len(srt)):
            texts.append(srt[i]['text'])

            ###
        str_list = " ".join(texts)

        doc = str_list

        ###
        okt = Okt()

        tokenized_doc = okt.pos(doc)
        tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

        ###
        n_gram_range = (2, 3)

        count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
        candidates = count.get_feature_names_out()

        ###
        model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
        doc_embedding = model.encode([doc])
        candidate_embeddings = model.encode(candidates)

        def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

            # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
            word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

            # 각 키워드들 간의 유사도
            word_similarity = cosine_similarity(candidate_embeddings)

            # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
            # 만약, 2번 문서가 가장 유사도가 높았다면
            # keywords_idx = [2]
            keywords_idx = [np.argmax(word_doc_similarity)]

            # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
            # 만약, 2번 문서가 가장 유사도가 높았다면
            # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
            candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

            # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
            # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
            for _ in range(top_n - 1):
                candidate_similarities = word_doc_similarity[candidates_idx, :]
                target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

                # MMR을 계산
                mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
                mmr_idx = candidates_idx[np.argmax(mmr)]

                # keywords & candidates를 업데이트
                keywords_idx.append(mmr_idx)
                candidates_idx.remove(mmr_idx)

            return [words[idx] for idx in keywords_idx]

        result_mmr = mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7)

        """
        리뷰 1개 출력
        """
        print_review = tmp['review'][0]

        return render_template("moredata2.html",
                               subCount=content,
                               statistics=stats,
                               snippet=snippet,
                               app_data=app_data,
                               # followings are shoppingData_name,price,img
                               search_list=search_list,
                               search_list_src=search_list_src,
                               search_list_price=search_list_price,
                               len=1,
                               positive_result=positive_result,
                               negative_result=negative_result,
                               # followings are shoppingData for cheapest price
                               result_cheapest = result_cheapest,
                               # followings are for keyword(hashtag) data
                               result_mmr = result_mmr,
                               print_review = print_review
                               )
    else:
        return "Page Not Found"

def naver_shopping():


    return render_template("shopping.html",
                           search_list=search_list,
                           search_list_src=search_list_src,
                           search_list_price=search_list_price,
                           len=1)

@app.route('/youtube_search')
def youtube_search():
    return render_template('youtube_search.html', app_data=app_data)

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        # getting the video details like images and channelid
        search = request.form['search']
        data = YouTubeData(search)
        snippet = data.get_channel_details(search)
        return render_template('search_page.html', message=snippet, search=search, app_data=app_data)
    else:
        return render_template('youtube_search.html', app_data=app_data)

@app.route('/get_more/<channelId>/<search>/<videoid>', methods=['GET', 'POST'])
def get_more(channelId, search, videoid):
    if request.method == 'GET':
        data = YouTubeData(search)
        content = data.get_channel_stats(channelId)

        snippet = data.get_videoDetails(videoid)

        stats = data.get_statistics(videoid)

        return render_template("moredata.html", subCount=content, statistics=stats, snippet=snippet, app_data=app_data)
    else:
        return "Page Not Found"


@app.route('/contact')
def contact():
    return render_template('contact.html', app_data=app_data)


if __name__ == '__main__':
    app.run(debug=DEVELOPMENT_ENV)