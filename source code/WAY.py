import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from PyQt5.QtWebEngineWidgets import QWebEngineView

from pathlib import Path
import geopandas as gpd # GeoPandas(지오판다스)
from glob import glob
from natsort import natsorted
import os
import rasterio
import shutil
import wayimpute
import pandas as pd
from pylab import plt

import io
from folium import Marker, Circle, folium
from jinja2 import Template

from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression # 로지스틱회귀
from xgboost import XGBClassifier # 엑스지부스트
from lightgbm import LGBMClassifier # 라이트그레디언트부스팅
from sklearn.preprocessing import LabelEncoder

from osgeo import ogr
import numpy as np
# import geopandas as gpd
import shapely.geometry

ogr.UseExceptions()

def erase_shapes(to_erase, eraser, out_file):
    feat1 = ogr.Open(to_erase)
    feat2 = ogr.Open(eraser)
    feat1Layer = feat1.GetLayer()
    feat2Layer = feat2.GetLayer()

    driver = ogr.GetDriverByName('ESRI Shapefile')
    outDataSource = driver.CreateDataSource(out_file)
    srs = feat1Layer.GetSpatialRef()
    outLayer = outDataSource.CreateLayer('', srs, ogr.wkbPolygon)

    out_ds = feat1Layer.Erase(feat2Layer, outLayer)
    out_ds = None

#임의 좌표 생성 함수
def random_points_in_gdf(gdf, size, overestimate=2):
    polygon = gdf['geometry'].unary_union # 합집합(union) 도형
    min_x, min_y, max_x, max_y = polygon.bounds # 폴리곤 영역
    ratio = polygon.area / polygon.envelope.area # 면적 비율 = 폴리곤 면적 / 경계상자 면적

    # 임의 좌표 생성: 포인트 개수(size) / 면적 비율(ratio) * 과대 산정(overestimate)
    samples = np.random.uniform((min_x, min_y), (max_x, max_y), (int(size / ratio * overestimate), 2))
    multipoint = shapely.geometry.MultiPoint(samples)
    multipoint = multipoint.intersection(polygon)
    samples = np.array(multipoint)

    points = samples[np.random.choice(len(samples), size)]
    df = pd.DataFrame(points, columns=['lon', 'lat'])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))


###############################
# python실행파일 디렉토리
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("gui\\qtui.ui")
form_class = uic.loadUiType(form)[0]
################################




# emit 형태로 쏴야 좌표에 나타남. Qthread 에서의 emit은 작동안해서 Qobject로 구현 !@#
class Provider(QObject):
    send_LatLon_r = pyqtSignal(float, float)
    send_LatLon_b = pyqtSignal(float, float)

    train = pd.DataFrame()
    pre_shape0_o = 0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(interval=1000)
        self._timer.timeout.connect(self.generate_coordinate)

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def generate_coordinate(self):
        # red
        for _, row in self.train[:self.pre_shape0_o].iterrows():
            # self.add_marker(row['lat'], row['lon'])
            self.send_LatLon_r.emit(row['lat'], row['lon'])

        # black
        for _, row in self.train[self.pre_shape0_o:].iterrows():
            self.send_LatLon_b.emit(row['lat'], row['lon'])

        self.stop()

class MThread(QThread):
    send_distr = pyqtSignal(np.ndarray)
    send_text = pyqtSignal(str)    # textedit 에 추가할 텍스트
    send_count = pyqtSignal(int)   # 모델별 평가이미지 출력할건데 페이지로 활용
    send_ROCpath = pyqtSignal(str) # QPixmap에 추가할 이미지 경로
    send_FIpath = pyqtSignal(str)  # QPixmap에 추가할 이미지 경로
    send_status = pyqtSignal(int)  # 버튼 상태(사용, 불가능)여부
    send_Mname = pyqtSignal(str)

    CLASS_MAP = { } # 사용되는 모델
    ascs = []       # ascs 파일 목록
    path = " "      # py 파일이 존재경로
    count = 0       # 현재 페이지
    count_all = 0   # CLASS_MAP 에서 모델의 개수 -> 시각화 개수
    e_path_li = []  # roc curve, 변수 중요도 경로를 저장하는 용도
    fname = ""      # getOpenFileName 받는 객체
    train_vec = pd.DataFrame() # 모델링에 사용되는 객체
    col_num = -1
    Mnames = []     # Model names
    userpath = os.path.expanduser('~') + '\\Desktop\\'

    def __init__(self):
        super().__init__()


    def incoder(self):
        le = LabelEncoder()
        col_names = self.train_vec.columns.values.tolist()
        col_name = col_names[self.col_num]
        self.train_vec[col_name] = le.fit_transform(self.train_vec[col_name])
        self.train_vec.to_csv(self.userpath + 'WAY\\INPUT\\TRAIN_VEC.csv', index=False) # 변경사항 저장

    def next(self):
        self.count += 1
        self.send_Mname.emit(self.Mnames[self.count])

        self.send_ROCpath.emit(self.e_path_li[self.count] + '\\_ROC Curve.png')
        self.send_FIpath.emit(self.e_path_li[self.count] + '\\_Feature importances.png')

        if self.count == self.count_all:
            self.send_status.emit(1)
        else: self.send_status.emit(3)


    def prev(self):
        self.count -= 1
        self.send_Mname.emit(self.Mnames[self.count])

        self.send_ROCpath.emit(self.e_path_li[self.count] + '\\_ROC Curve.png')
        self.send_FIpath.emit(self.e_path_li[self.count] + '\\_Feature importances.png')
        if self.count == 0 :
            self.send_status.emit(2)
        else: self.send_status.emit(3)

    def run(self):
        self.send_text.emit( "사용된 모델: " + " ".join(list(self.CLASS_MAP.keys()))) # 사용된 모델 출력
        self.count_all = len(self.CLASS_MAP) - 1
        self.Mnames = list(self.CLASS_MAP.keys())

        # self.train_vec = pd.read_csv(self.fname) # csv 파일 읽어들일때 read_csv 하므로 주석처리
        self.send_text.emit( self.fname + " 로부터 읽어들임...")
        try: self.train_vec.drop(['lat','lon','geometry'], axis=1, inplace=True) #
        except KeyError: self.send_text.emit("drop 실패")

        # 모델링 시작 & INPUT 데이터 복사
        for f in self.ascs:
            shutil.copy(f, self.userpath + '\\WAY\\INPUT')

        self.send_text.emit(self.userpath + "\\WAY\\INPUT 으로 선택된 환경변수를 복사...")

        raster_features = [] # 컬럼제거 기능에 대비하여 glob이 아닌 리스트로 대체
        pp = self.userpath + '\\WAY\\INPUT\\'
        for col_name in self.train_vec.iloc[:, 1:].columns.values.tolist():
            raster_features.append(pp + col_name + ".tif")

        self.send_text.emit(str(len(raster_features))+ '개 래스터 특징')
        # raster_features = self.train_vec.columns[1:].values.tolist()

        train_xs, train_y = self.train_vec.iloc[:, 1:].values, self.train_vec.iloc[:, 0].values
        target_xs, raster_info = wayimpute.load_targets(raster_features)
        self.send_text.emit("입력 데이터 행의 개수 : " + str(train_xs.shape[0]))
        self.send_text.emit("정답 데이터(CLASS) 행의 개수 : " + str(train_y.shape[0]))

        # 독립변수만 따로 모아서 리스트 형태로 저장 -> 나중에 특성 중요도 그래프 그리기 위해 필요함
        col = self.train_vec.columns[1:]
        columns = list(col)

        # 히트맵 그리기 위한 데이터 지정
        # heatmap_data = self.train_vec[col]

        # ML 분류기 딕셔너리: 이름, (모델)
        # 딕셔너리 키(key)와 값(value)을 한 쌍으로 저장

        self.send_text.emit("\n" * 2)
        self.send_text.emit("모델링 시작")

        i = 0
        for name, (model) in list(self.CLASS_MAP.items()):
            try:
                e_path_part = '\\WAY\\OUTPUT\\' + name + '-IMAGES'
                e_path = self.userpath + e_path_part # e_path : evaluate_path 약어
                self.e_path_li.append(e_path)
                os.mkdir(e_path)
            except: pass
            # roc, 등 평가지표가 파일에 저장됨.

            FI = np.array([]) # if FI 하기 위해 선언, FI = Feature Importances
            if name in ['BAG','Maxent']: # 특성 중요도 안나오는 모델들
                Kfold, AccScore, clf_report, matrix, RA_score = wayimpute.evaluate_clf(model, train_xs, train_y, name, e_path,k=5, test_size=0.2, scoring="f1_weighted",
                                                                            feature_names=columns)
            else:
                Kfold, AccScore, clf_report, matrix, RA_score, FI = wayimpute.evaluate_clf(model, train_xs, train_y, name, e_path,k=5, test_size=0.2, scoring="f1_weighted",
                                                                            feature_names=columns)

            if i==0: # 첫번째 모델 지표만 표기
                self.send_ROCpath.emit(e_path + '\\_ROC Curve.png')
                self.send_FIpath.emit(e_path + '\\_Feature importances.png')
                i += 1
            self.send_text.emit(name)
            if Kfold: self.send_text.emit(Kfold)
            self.send_text.emit("Accuracy Score(정확도점수): %.4f" % AccScore)
            self.send_text.emit("Classification report(평가 지표)")
            pp = str(clf_report).split('\n')
            for line in pp[:-3]:
                self.send_text.emit('     ' + line)
            for line in pp[-3:]:
                self.send_text.emit(line)
            self.send_text.emit("Confussion matrix(혼동 행렬)")
            self.send_text.emit(str(matrix))
            self.send_text.emit('ROC AUC(Roc 값): %.4f'% RA_score)

            if FI.any(): # FI 존재시 특성 중요도 출력
                self.send_text.emit("Feature importances(특성 중요도)")
                for f, imp in zip(columns, FI):
                    self.send_text.emit("%20s: %s" % (f, round(imp * 100, 1)))
            else: self.send_text.emit("특성 중요도를 추출할 수 없는 모델입니다.")


            # 공간 예측(spatial prediction)
            model.fit(train_xs, train_y)
            wayimpute.impute(target_xs, model, raster_info, outdir=self.userpath + 'WAY/OUTPUT/' + name + '-IMAGES',
                             class_prob=True, certainty=True)
            self.send_text.emit("*" *40)
        distr_sum = 0
        for path in self.e_path_li:
            distr_sum += rasterio.open(path + '\\probability_1.0.tif').read(1)

            # distr_averaged.append(rasterio.open(path + '\\probability_1.0.tif'))
        self.send_distr.emit(distr_sum/len(self.e_path_li))
        # plotit(distr_averaged, "Joshua Tree Range, averaged", cmap="Greens")

        self.send_Mname.emit(self.Mnames[0])
        self.send_text.emit("결과물들이 " + self.path + "\\OUTPUT 폴더에 저장됨")
        self.send_status.emit(2)
        # plotit(distr_averaged[100:150, 100:150], "Joshua Tree National Park Suitability", cmap="Greens")


class MyWindow(QMainWindow, form_class):
    pre_shape0 = 0 # presence.shape[0], 출현좌표 쉐잎
    def __init__(self):
        super().__init__()

        self.setupUi(self)
        self.modeling_thread = MThread()
        self.modeling_thread.path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

        self.Ob_thread = Provider()
        self.Ob_thread.send_LatLon_r.connect(self.add_marker_r)
        self.Ob_thread.send_LatLon_b.connect(self.add_marker_b)


        # 시그널
        self.modeling_thread.send_distr.connect(self.plot)
        self.modeling_thread.send_text.connect(self.TextEdit)
        self.modeling_thread.send_ROCpath.connect(self.show_ROC)
        self.modeling_thread.send_FIpath.connect(self.show_Feature_Importances)
        self.modeling_thread.send_status.connect(self.status)
        self.modeling_thread.send_Mname.connect(self.Mname_update)
        # self.modeling_thread.send_MT.connect(self.MakeTable)
        # self.modeling_thread.send_LatLon.connect(self.add_marker)

        os.chdir(self.modeling_thread.path)
        self.userpath = os.path.expanduser('~') + '\\Desktop\\'
        os.makedirs(self.userpath + 'WAY', exist_ok=True)  # 결과물 저장될 폴더 생성

        # 돌릴때마다 input output 폴더 초기화
        if os.path.exists(self.userpath + 'WAY\\INPUT'):
            shutil.rmtree(self.userpath + 'WAY\\INPUT')
        os.makedirs(self.userpath + 'WAY\\INPUT', exist_ok=True)  # 작업 디렉토리에 'INPUT(입력) 폴더 생성'

        if os.path.exists(self.userpath + 'WAY\\OUTPUT'):
            shutil.rmtree(self.userpath + 'WAY\\OUTPUT')
        os.makedirs(self.userpath + 'WAY\\OUTPUT', exist_ok=True)  # 작업 디렉토리에 'OUTPUT(출력) 폴더 생성'

        # 버튼 클릭
        self.AscButton.clicked.connect(self.filesSelect)
        # self.TifButton.clicked.connect(self.filesSelect,2)
        # self.CsvButton.clicked.connect(self.filesSelect,3)
        self.SamplingButton.clicked.connect(self.fileSelect)
        self.CsvButton.clicked.connect(self.fileSelect)
        self.ModelingButton.clicked.connect(self.Modeling)
        self.btn_next.clicked.connect(self.next)
        self.btn_prev.clicked.connect(self.prev)
        self.IncoderBtn.clicked.connect(self.incoder)
        self.ColRM_btn.clicked.connect(self.ColRM)
        # self.ReSamBtn.clicked.connect(self.ReSampling) # "비출현 데이터 생성 버튼" 비활성화로 인한 주석처리

        # 시작전 버튼 활성화 세팅
        self.btn_next.setEnabled(False)
        self.btn_prev.setEnabled(False)

        self.table.setRowCount(0)
        self.table.setColumnCount(0)

    def Mname_update(self,str):
        self.label.setText(str)

    def ColRM(self): #선택한 컬럼 제거
        self.status(10)
        col_num = self.table.currentColumn()
        col_names = self.modeling_thread.train_vec.columns.values.tolist()
        col_name = col_names[col_num]
        self.modeling_thread.train_vec.drop(col_name,axis=1,inplace=True)
        self.modeling_thread.train_vec.to_csv(self.userpath + 'WAY\\INPUT\\TRAIN_VEC.csv', index=False)
        # self.modeling_thread.train_vec = pd.read_csv('WAY\\INPUT\\TRAIN_VEC.csv')

        self.table.clear()
        row, col = self.modeling_thread.train_vec.shape
        self.table.setRowCount(row)
        self.table.setColumnCount(col)
        self.table.setHorizontalHeaderLabels(self.modeling_thread.train_vec.columns.values.tolist())
        for i in range(row):
            for j in range(col):
                self.table.setItem(i, j, QTableWidgetItem(str(self.modeling_thread.train_vec.iloc[i, j])))
        self.status(11)

    def status(self,status_num):
        if status_num == 1: # 1 0
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(False)
        elif status_num == 2: # 0 1
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(True)
        elif status_num == 3: # 1 1
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(True)

        elif status_num == 10: # 10부터는 main페이지 버튼
            self.SamplingButton.setEnabled(False)
            self.AscButton.setEnabled(False)
            self.CsvButton.setEnabled(False)
            self.ColRM_btn.setEnabled(False)
            self.IncoderBtn.setEnabled(False)
            # self.ReSamBtn.setEnabled(False) # "비출현 데이터 생성 버튼" 비활성화로 인한 주석처리
        elif status_num == 11: # 10부터는 main페이지 버튼
            self.SamplingButton.setEnabled(True)
            self.AscButton.setEnabled(True)
            self.CsvButton.setEnabled(True)
            self.ColRM_btn.setEnabled(True)
            self.IncoderBtn.setEnabled(True)
            # self.ReSamBtn.setEnabled(True) # "비출현 데이터 생성 버튼" 비활성화로 인한 주석처리


    def next(self):
        self.modeling_thread.next()

    def prev(self):
        self.modeling_thread.prev()

    def TextEdit(self, text):
        self.textEdit.append(text)

    def Modeling(self):
        if self.RFCB.isChecked():
            self.modeling_thread.CLASS_MAP['RF'] = (RandomForestClassifier())
        if self.ETCB.isChecked():
            self.modeling_thread.CLASS_MAP['ET'] = (ExtraTreesClassifier())
        if self.AdaCB.isChecked():
            self.modeling_thread.CLASS_MAP['ADA'] = (AdaBoostClassifier())
        if self.BaggingCB.isChecked():
            self.modeling_thread.CLASS_MAP['BAG'] = (BaggingClassifier())
        if self.GBCB.isChecked():
            self.modeling_thread.CLASS_MAP['GRA'] = (GradientBoostingClassifier())
        if self.XGBCB.isChecked():
            self.modeling_thread.CLASS_MAP['XGB'] = (XGBClassifier())
        if self.LGBMCB.isChecked():
            self.modeling_thread.CLASS_MAP['LGBM'] = (LGBMClassifier())
        if self.MaxentCB.isChecked():
            self.modeling_thread.CLASS_MAP['Maxent'] = (LogisticRegression())

        self.modeling_thread.start() # run 도중 send_distr.emit으로 인해 plot 슬롯으로 넘어감

    def plot(self, distr_averaged):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        ax = self.figure.add_subplot(111)
        img = ax.imshow(distr_averaged, cmap="magma", interpolation='nearest')
        self.figure.colorbar(img)
        # self.figure.set_title('PyQt Matplotlib Example')
        ax.set_title('Range Average')
        self.PltLayout.addWidget(self.canvas)
        self.canvas.draw()

    def show_ROC(self,ROCpath): # 2번째 페이지에서 ROC 이미지 출력
        pixmap = QPixmap(ROCpath)
        # pixmap = pixmap.scaledToWidth(600)
        self.label_1.setPixmap(pixmap)

    def show_Feature_Importances(self,FIpath): # 2번째 페이지에서 Feature_Importance 이미지 출력
        pixmap = QPixmap(FIpath)
        self.label_2.setPixmap(pixmap)
        # self.label_2.setScaledContents(True)

    def filesSelect(self):
        self.status(10)
        fsname = QFileDialog.getExistingDirectory(self,'환경변수 선택')
        p = str(Path(str(fsname)).resolve())  # os-agnostic absolute path p = \
        files = natsorted(glob(os.path.join(p, '*.*')))  # dir
        self.modeling_thread.ascs = [x for x in files if x.endswith('.tif')]
        # 출현좌표 읽기
        presence = gpd.GeoDataFrame.from_file(self.coor_dir[0])
        presence_bfr = presence.to_crs(5179)  # 투영좌표계
        presence_bfr['geometry'] = presence_bfr.geometry.buffer(100)
        presence_bfr = presence_bfr.to_crs(4326)  # 지리좌표계

        # difference = 행정구역 - 출현지점 버퍼
        total_area = gpd.GeoDataFrame.from_file(self.modeling_thread.path + '\\ADM_KOR\\ADM_KOR.gpkg')
        res_difference = total_area.overlay(presence_bfr, how='difference')

        # 임의지점 생성 함수
        self.pre_shape0 = presence.shape[0] # folium 좌표 색깔 구분을 위해 선언

        absence = random_points_in_gdf(res_difference, presence.shape[0])
        absence['CLASS'] = 0

        # 출현지점 + 비출현지점 저장
        gdf = presence.append(absence, ignore_index=True)
        gdf.crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        gdf.to_file(self.userpath + 'WAY\\INPUT\\Occurrences.gpkg', driver='GPKG', name='Occurrences')

        # 포인트 샘플링
        coord_list = [(x, y) for x, y in zip(gdf['geometry'].x, gdf['geometry'].y)]
        for file in self.modeling_thread.ascs:
            src = rasterio.open(file)  # 파일 읽기
            gdf[Path(file).stem] = [x for x in src.sample(coord_list)]  # 포인트 샘플링
            gdf[Path(file).stem] = gdf[Path(file).stem].astype('float64')

        # 포인트 샘플링 시 NODATA 오류 탐색: 생물기후 변수가 같은 값
        columns = gdf.columns.difference(['CLASS', 'lon', 'lat', 'geometry'])
        nodata_df = gdf[gdf[columns].nunique(axis=1) == 1]
        self.pre_shape0 = self.pre_shape0 - nodata_df[nodata_df['CLASS'] ==1 ].shape[0]
        gdf.drop(nodata_df.index,inplace= True)


        # gdf로 train_vec.csv 형성( gdf는 지오판다스 기반이라 사이킷런에 넣을수 없다 )
        gdf.to_csv(self.userpath + 'WAY\\INPUT\\TRAIN_VEC.csv',index=False)
        # gdf[gdf.columns.difference(['geometry'])].to_csv('WAY\\INPUT\\TRAIN_VEC.csv', index=False)
        # diff['geometry'] -> 리샘플링에 필요하므로 주석처리

        self.modeling_thread.train_vec = pd.read_csv(self.userpath + 'WAY\\INPUT\\TRAIN_VEC.csv') # 리샘플링 -> 모델링이므로 read_csv

        # elif num == 2:
        #     self.tifs = [x for x in files if x.endswith('.tif')]
        self.status(11)


    def fileSelect(self): # 파일경로 한개만 가져옴
        self.status(10)
        self.fname = QFileDialog.getOpenFileName(self) # shp(샘플링 지점), csv(csv 파일) 버튼으로 사용
        if self.fname[0].endswith(".csv"): # csv 파일일 경우 MThread의 run에 사용
            self.modeling_thread.fname = self.fname[0]
            self.modeling_thread.train_vec = pd.read_csv(self.modeling_thread.fname)
            self.modeling_thread.train_vec.to_csv(self.userpath + 'WAY\\INPUT\\TRAIN_VEC.csv', index=False)
            # self.MakeTable(True)

            row, col = self.modeling_thread.train_vec.shape
            self.table.setRowCount(row)
            self.table.setColumnCount(col)
            self.table.setHorizontalHeaderLabels(self.modeling_thread.train_vec.columns.values.tolist())
            for i in range(row):
                for j in range(col):
                    self.table.setItem(i, j, QTableWidgetItem(str(self.modeling_thread.train_vec.iloc[i, j])))
            self.show_folium()

        else: self.coor_dir = self.fname # shp 파일일경우 지정
        self.status(11)

    def show_folium(self):
        # try: self.map_layout.itemAt(0).widget().deleteLater()  # 2번째 이상부터 지도 초기화
        # except AttributeError: pass  # 처음 나타내는 지도면 pass
        lat_mean = self.modeling_thread.train_vec.lat.mean()
        lon_mean = self.modeling_thread.train_vec.lon.mean()
        mean_coordinate = (lat_mean, lon_mean)
        self.map = folium.Map(location=mean_coordinate, zoom_start=7)

        data = io.BytesIO()
        self.map.save(data, close_file=False)
        self.map_view = QWebEngineView()
        self.map_view.setHtml(data.getvalue().decode())
        self.map_layout.addWidget(self.map_view)

        self.Ob_thread.train = self.modeling_thread.train_vec.copy()
        self.Ob_thread.pre_shape0_o = self.pre_shape0
        self.Ob_thread.start()


    def add_marker_r(self, latitude, longitude):
        js = Template(
            """
        L.circle([{{latitude}}, {{longitude}}], {
                    "color": "#ff0000"
            }
         ).addTo({{map}});
        """
        ).render(map=self.map.get_name(), latitude=latitude, longitude=longitude)
        self.map_view.page().runJavaScript(js)

    def add_marker_b(self, latitude, longitude):
        js = Template(
            """
        L.circle([{{latitude}}, {{longitude}}], {
                    "color": "#000000"
            }
         ).addTo({{map}});
        """
        ).render(map=self.map.get_name(), latitude=latitude, longitude=longitude)
        self.map_view.page().runJavaScript(js)

    def incoder(self):
        self.status(10)
        self.modeling_thread.col_num = self.table.currentColumn()
        self.modeling_thread.incoder() #
        # self.table.clear()

        row, col = self.modeling_thread.train_vec.shape
        self.table.setRowCount(row)
        self.table.setColumnCount(col)
        self.table.setHorizontalHeaderLabels(self.modeling_thread.train_vec.columns.values.tolist())
        for i in range(row):
            for j in range(col):
                self.table.setItem(i, j, QTableWidgetItem(str(self.modeling_thread.train_vec.iloc[i, j])))
        # self.modeling_thread.send_MT = False
        self.status(11)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()