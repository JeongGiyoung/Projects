# [프로젝트 일지]

## 개발 구조
<p align = "center">
<img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg"  width = "600" hight = "480" >
</p>


[프로젝트 개요]

# 빅데이터 과정 파이널 프로젝트
  * 제목 : S.YOL.C(유튜브 영상 내 여성 뷰티제품 실시간 정보 제공 서비스)
    - Smart + YOLO + Comsumption
  * 서비스 목표 : 유튜브로 제품 리뷰를 보고 다시 네이버 쇼핑몰에서 상품 정보 찾아보는 번거로움 해결을 목표. 
  * 주요기술 : 
    - 언어 : python
    - IDE : 아나콘다, 구글 colab
      + 객체 탐지(objection dectection)
        * YOLOv5 : 유튜브 영상 내 뷰티 제품을 YOLOv5(객체 탐지 모델)로 인식 
      + crawling 
        * selenium : 상품정보를 네이버 쇼핑에서 scrapping
        * beautifulSoup 
      + NLP
        * mecab, koNLPy
        * 네이버 쇼핑 리뷰 100개 감성분석(긍부정 판별 및 비율)
    - web-framwork 
      + 언어 : flask(python), html(+bootstrap)
      + IDE : pycharm

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fca2ctf%2FbtrGyOxPxif%2FEJCdDdks6EwPzu7FeeUXF0%2Fimg.png">

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbCtFtR%2FbtrGFKiqwAT%2FdWapyemFwn6iqkQJBw4qAk%2Fimg.png">


<hr>

# [프로젝트 일지]

# # 2022-07-07(목)
  * ppt 수정
    - 마지막 점검

# # 2022-07-06(수)
  * ppt 수정
    - 리뷰 텍스트 1개 출력

# # 2022-07-05(화)
  * ppt 수정
    - 쓰인 기술 소개 정리
    - 웹캠 시현 준비

# # 2022-07-02(토)
  * 멘토링

# # 2022-07-01(금)
  * 중간발표
  * feedback 적용

# # 2022-06-30(목)
  * html 버튼 스피너
  * 추천시스템 추가
  * ppt(리허설) 준비

# # 2022-06-29(수)

- 해쉬태그 뽑기

# # 2022-06-28(화)

- easyocr + opencv
    - innisfree 제품

# # 2022-06-25(토)

- 오류 수정

# # 2022-06-24(금)

- 웹서비스에 구현(on flask)
    - 상품정보
    - 리뷰 긍부정 비율
- ppt 수정

# # 2022-06-23(목)

- 중간점검 ppt
- 웹서비스에 구현(on flask) (미완료)

# # 2022-06-22(수)

- OCR로 상표 읽기
- 중간점검 ppt


# # 2022-06-21(화)

- 네이버 쇼핑몰 크롤링
- mecab 설치
- 네이버 쇼핑몰 리뷰 감성분석

# # 2022-06-18(토)

- 멘토링

# # 2022-06-17(금)

- 네이버 쇼핑몰 리뷰 크롤링(미완료)

# # 2022-06-16(목)

- 데이터 1000개 수집(2개 객체 인식) 및 bounding box
    
    [상품 2개 탐지 일지(VodanaStraightener)](https://www.notion.so/2-VodanaStraightener-df3f6118a5a8403d8566fb4d09787e44)
    
- DL service on Flask
    - https://github.com/robmarkcole/yolov5-flask

# # 2022-06-15(수)

- Youtube 검색 페이지를 프로젝트 페이지와 합치기
- opencv 전처리

# # 2022-06-14(화)

- OpenCV + tesseract 로 OCR(문자 판독)
- OpenCV + tesseract 로 자동차 번호판 판독
- Flask로 Youtube 검색 결과 출력

# # 2022-06-10(금)

- Flask로 네이버 쇼핑몰 검색 결과와 영상 같이 나오게 꾸미기(미완료)
    - → 서비스 페이지 형태를 수정
        - Youtube 검색 결과가 나오는 페이지, 재생을 요구한 영상 페이지 나누기(on Flask)


# # 2022-06-09(목)

- Flask로 웹캠 화면 출력
- YOLOv5 웹캠으로 객체 인식


# # 2022-06-08(수)

- Flask로 웹크롤링(오늘의유머) 공부
- Flask로 네이버 쇼핑몰 검색

# # 2022-06-7(화)
- 6월3일에 소규모 데이터로 구현했던 테스트 결과물을 web(Django에 구현)


# # 2022-06-03(금)

- 소규모 테스트 데이터로 시현
    - ****ALL NIGHTER SETTING SPRAY 118ml****
    - 크롬 확장프로그램으로 Image Downloader
    
    [Google Colaboratory](https://colab.research.google.com/drive/1XO_dWA0fi81lJQSq_jObpFvrsdhYR67d?usp=sharing)