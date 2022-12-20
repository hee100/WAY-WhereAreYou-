
<img src='https://user-images.githubusercontent.com/95641633/208577817-e439ef66-b778-4cf2-90c5-92e080806662.png' width="30%" height="40%"> ![BigLeader_logo](https://user-images.githubusercontent.com/95641633/208577847-d4472aca-8852-418b-9cda-cbddb74a6ec8.png)


# 소개
종 분포 모델링(SDM: Species Distribution Modeling)이란, 컴퓨터 알고리즘을 사용하여 지리적 시공간에 걸쳐 종의 분포를 예측하는 모델링입니다.

WAY(WhereAreYou)는 [파이썬 개발자를 위한 종 분포 모델링](https://github.com/osgeokr/SDM4PyDev) 을 기반으로 만들어진 GUI 형태의 SDM 프로그램으로, 현재 pilot 단계에 있습니다. 
프로그램 WAY를 통해 파이썬 코딩, 머신러닝 등의 사전지식 없이 SDM을 활용할수 있도록 설계했습니다. 

# 사용방법
* 본 예시에서는 레퍼지토리에 기재된 TestData를 사용하였습니다. <br/><br/>

폴더중 WAY1.0 폴더를 열어 돋보기 아이콘의 WAY.exe 파일을 열어줍니다.
![image](https://user-images.githubusercontent.com/95641633/208608993-57e7a6cb-80b3-4bec-a64b-289add2e8ec8.png)<br/><br/><br/>          
  
아래와 같은 화면이 나타납니다. 
![image](https://user-images.githubusercontent.com/95641633/208609156-87a39a82-f763-4915-8b40-4f3a13235d62.png)<br/><br/><br/>


1. 위치 좌표 버튼을 클릭하여 shp, gpkg등의 원하는 종의 좌표 데이터를 불러옵니다.
![image](https://user-images.githubusercontent.com/95641633/208610459-f0f33603-e733-4083-8ee0-108ba9800a0b.png)<br/><br/><br/>

바탕화면에 "WAY"라는 새로운 폴더가 나타납니다.    
![image](https://user-images.githubusercontent.com/95641633/208611891-4058a46e-7aae-4496-b732-bf1b58ab2e88.png)<br/><br/>
WAY 폴더안에 두개의 빈 폴더가 있습니다.
INPUT 폴더에는 리샘플링이 완료된 좌표계와 최종 모델링때 사용된 환경변수, 모델링을 위해 만들어진 csv 파일이 형성됩니다.
OUTPUT 폴더에는 최종 결과물들이 형성 됩니다.
![image](https://user-images.githubusercontent.com/95641633/208613087-ba2d386a-c83d-445e-b133-b3e4e7015858.png)<br/><br/><br/>

2. 환경변수 버튼을 클릭하여 환경변수 폴더를 가져옵니다. ( 폴더 선택시 선택된 폴더 안의 모든 tif 파일을 가져옵니다. )
![image](https://user-images.githubusercontent.com/95641633/208611253-6061dffe-bb74-4de2-92ce-e2c0afcfef73.png)<br/>
리샘플링이 완료돤 좌표계(Occurrences.gpkg)와 모델링에 사용될 CSV파일(TRAIN_VEC.csv)가 형성됩니다.
![image](https://user-images.githubusercontent.com/95641633/208615190-ff6ebae2-7feb-43af-813d-4c70e80d7d27.png)<br/><br/>

3. csv파일 버튼을 클릭해 TRAIN_VEC.csv을 선택해줍니다.
![image](https://user-images.githubusercontent.com/95641633/208614980-cd4d8af3-24dd-4828-a9c0-b3bc44624fd8.png)<br/><br/>
좌측에 TRAIN_VEC.csv의 내용을 담은 테이블이 나타나며 우측에는 대한민국 지도안의 위치좌표가 나타납니다. ( 빨간좌표가 출현좌표이며 검은좌표는 비출현좌표로, 리샘플링된 좌표입니다.)
![image](https://user-images.githubusercontent.com/95641633/208615339-434a9dcf-a5e1-4f1e-a1bc-7c584ca837da.png)<br/><br/><br/>


하나의 컬럼을 선택후 '컬럼 제거' 버튼을 누르면
![image](https://user-images.githubusercontent.com/95641633/208616161-e458078c-bcef-4baa-a0d0-d61975f733c0.png)<br/><br/>
해당 컬럼이 제거된 TRAINV_VEC.csv 파일과 테이블이 나타나며, 모델링시 해당 컬럼은 제외한채로 진행됩니다.<br/><br/><br/>

상단의 Result 탭으로 이동후 원하는 모델에 체크하여 '모델링시작' 버튼을 누르면 모델링이 시작됩니다.
![image](https://user-images.githubusercontent.com/95641633/208617532-9d931566-e2c4-4b64-8e73-5919550e374a.png)<br/><br/>

모델 결과를 이미지로 출력하며 이전 / 다음 버튼을 통해 다른 모델의 결과를 확인합니다.<br/>
하단 텍스트 에디터에 각 모델마다의 결과물을 텍스트로 출력합니다. 
![image](https://user-images.githubusercontent.com/95641633/208618539-8ecfce9a-76ce-407c-88b5-fd60df2cc30e.png)<br/><br/>

OUTPUT 폴더에 결과물들이 생성됩니다.
![image](https://user-images.githubusercontent.com/95641633/208618348-528184e7-8215-4aa5-857e-20947890b4a5.png)<br/><br/>

Mapping 탭에서 모든 모델의 평균 결과값을 보여줍니다.
![image](https://user-images.githubusercontent.com/95641633/208619471-e246b9cf-2aeb-418e-9460-881e66e31f54.png)




### 주의 사항
- 환경변수 파일 형식은 tif 형식만 허용됩니다.
- csv파일 선택후 인터넷 문제로 좌표계가 잘못찍힐 수 있습니다. 하지만 모델링에는 문제가 없습니다.
- lon,lat,geometry 컬럼 제거시 모델링이 되지 않습니다.
- 인코딩 버튼은 구현중에 있습니다.











# Data
- BIOCLIM 폴더: 한반도 영역(33°∼43°N, 124°∼132°E)의 1970~2000년 평균 19종 생물기후 변수(Bioclimatic variables) [데이터 출처](https://www.worldclim.org/data/bioclim.html)
- Zosterops_japonicus.gpkg: 동박새(Warbling white-eye) 출현(presence) 좌표 [데이터 출처](https://plugins.qgis.org/plugins/qgisgbifapi/)
- ADM_KOR.gpkg: 행정구역 시군구 경계 [데이터 출처](http://data.nsdi.go.kr/dataset/20180927ds0058)


## 팀, 멘토소개
팀명 'T-mirum'은 다양한 분야에 깊은 지식을 가지는 T자형 인재 + 라틴어 놀라다 'mirum'의 합성어 입니다. 저희 T-mirum은 22년 빅리더 AI 아카데미에 참가하여 멘토 유병혁 과장님이 주관하는 종 분포 모델링 프로젝트에 참여하여 프로그램 WAY를 개발했습니다.

- T-mirum
  - 공희배 (경남대학교 전자공학과)
  - 전영웅 (동서울대학교 소프트웨어학과)
  - 이은주 (단국대학교 수학과; 컴퓨터공학과)
  - 김진규 (경남대학교 경제금융학과; 빅데이터AI학과)
  - 김민기 (서울과학기술대학교 컴퓨터공학과)

- 멘토
  - 배진익 (PT Mobility Doctor Indonesia)
  - 유병혁 (국립공원공단 사회가치혁신실 bhyu@knps.or.kr; OSGeo Charter Member; GGRS 기술블로거)
 
- 테스터
  - 수동, 김현, 박경식, 오충현, 조봉교, 진민화 (경상국립대학교 조경학과 대학원)
  - 안미연 (부산대학교 조경학과 대학원)
  - 윤성수 (국립생태원 생태정보팀)
  - 성정모 (국립공원공단 속리산국립공원사무소)

## With us
### [Pyinstaller](https://pyinstaller.org/en/stable/#)

