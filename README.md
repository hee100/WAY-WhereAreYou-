
![KNPS_logo](https://user-images.githubusercontent.com/95641633/208577817-e439ef66-b778-4cf2-90c5-92e080806662.png){: width="60%"",height="40%""}  ![BigLeader_logo](https://user-images.githubusercontent.com/95641633/208577847-d4472aca-8852-418b-9cda-cbddb74a6ec8.png)


# 소개
종 분포 모델링(SDM: Species Distribution Modeling)이란, 컴퓨터 알고리즘을 사용하여 지리적 시공간에 걸쳐 종의 분포를 예측하는 모델링입니다.

WAY(WhereAreYou)는 [파이썬 개발자를 위한 종 분포 모델링](https://github.com/osgeokr/SDM4PyDev) 을 기반으로 만들어진 GUI 형태의 SDM 프로그램으로, 현재 pilot 단계에 있습니다. 
프로그램 WAY를 통해 컴퓨터 알고리즘 사전지식 없이 SDM을 활용할수 있도록 설계했습니다. 

# 사용방법




# Data
- BIOCLIM 폴더: 한반도 영역(33°∼43°N, 124°∼132°E)의 1970~2000년 평균 19종 생물기후 변수(Bioclimatic variables) [데이터 출처](https://www.worldclim.org/data/bioclim.html)
- Zosterops_japonicus.gpkg: 동박새(Warbling white-eye) 출현(presence) 좌표 [데이터 출처](https://plugins.qgis.org/plugins/qgisgbifapi/)
- ADM_KOR.gpkg: 행정구역 시군구 경계 [데이터 출처](http://data.nsdi.go.kr/dataset/20180927ds0058)


# 팀, 멘토소개
팀명 'T-mirum'은 다양한 분야에 깊은 지식을 가지는 T자형 인재 + 라틴어 놀라다 'mirum'의 합성어 입니다. 저희 T-mirum은 

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

## exe 패키징
### [Pyinstaller](https://pyinstaller.org/en/stable/#)

