

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import warnings
warnings.simplefilter("ignore")
import logging

from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn import model_selection

logger = logging.getLogger("pyimpute") # pyimpute라는 모듈을 Logger형태로 만들기err = traceback.format_exc()

def impute(  # target 데이터와 scikit-learn 분류기, 예측한 값들을 GeoTiff 이미지로 export하는 함수
        target_xs,  # target_xs : 래스터 타겟의 데이터로 expl변수를 의미 (58800,19)의 2차원임
        clf,  # scikit-learn의 model_selection에서 만들어진 classifier 인스턴스 입력 RandomForest() 등등 모델이 들어가 있음
        raster_info,  # load_targets 함수에서 리턴된 raspip install --upgrade pipter_info는 딕셔너리 형태
        outdir='output',  # 출력할 결과물을 저장할 경로 지정
        linechunk=1000,  # 한 pass 작업당 처리할 라인수를 지정
        class_prob=True,  # boolean형. 각 클래스별로 확률 래스터를 만들까?
        certainty=True  # boolean형. 전반적인 분류 확실성 래스터를 만들까
):
    if not os.path.exists(outdir):  # 결과물 출력 파일 경로를 못찾겠으면
        os.makedirs(outdir)  # tiff 이미지 저장할 디렉토리라도 새로 만들어!!

    shape = raster_info["shape"]  # tiff 가로 세로 길이 shape로 지정하기
    # shape = (행길이, 열길이) 형태의 튜플
    # shape = (196, 300)

    profile = {
        "blockxsize": shape[1],  # 블록사이즈 shape 튜플의 열숫자로 지정
        "height": shape[0],  # 높이 shape 튜플의 행숫자로 지정
        "blockysize": 1,  # 1개 단위의 블록을 실제 몇개로 묶어서 지정 -> 무슨 의미지인지 잘 모르겠음
        "count": 1,  # count할 숫자 지정
        "crs": raster_info["crs"],  # 좌표계 종류를 나타내는 crs
        "driver": u"GTiff",  # GTiff 파일형태로 저장하도록 지시하는 driver
        "dtype": "int16",  # 원소 변수 저장할 타입은 16비트 int형
        "nodata": -32768,  # 데이터가 없는 원소 자리에 대체할 숫자 -> 숫자는 별 의미 없음
        "tiled": False,  # 격자가 있는 래스터 이미지인가?
        "transform": raster_info["transform"],  # affine 변환 행렬
        # (여기서는 load_targets에서 만들어진 raster_info 딕서너리의 transform 가져다가 사용)
        "width": shape[1],  # 너비 shape 튜플의 열숫자로 지정
    }  # 래스터파일을 여는데 필요한 사전정보가 담긴 profile 딕셔너리

    try:
        response_path = os.path.join(outdir, "responses.tif")  # responses tif 저장할 위치 지정 및 생성
        # profile 딕셔너리 정보를 가지고, 래스터 이미지의 dataset, response_ds 만들기
        # profile의 **은 위에 변수 디폴트값들도 있어서 유동적이라서
        response_ds = rasterio.open(response_path, "w", **profile)

        profile["dtype"] = "float32"
        if certainty:  # 분류확실성 래스터 만드는 옵션이 True면 -> 조금 더 찾아봐야하는 변수
            certainty_path = os.path.join(outdir, "certainty.tif")  # certainty tif 저장할 위치 지정 및 생성
            # 분류확실성 래스터의 데이터셋을 기록하도록 write 모드로 전환
            certainty_ds = rasterio.open(certainty_path, "w", **profile)

        class_dss = []
        if class_prob:  # 각 클래스별로 확률 래스터를 만드는 옵션이 True면 실행
            classes = list(clf.classes_)  # model의 출현, 비출현의 확률이 출력되는데 예) array([0., 1.]) 형태로 반환
            # output이미지에서 확인 가능
            class_paths = []
            for i, c in enumerate(classes):
                ods = os.path.join(outdir, "probability_%s.tif" % c)  # probability tif 저장할 위치 지정 및 생성
                class_paths.append(ods)
            for p in class_paths:
                # 각 클래스별로 확률 래스터를 기록하도록 write 모드로 전환
                class_dss.append(rasterio.open(p, "w", **profile))

        # Chunky logic
        if not linechunk:  # 한번에 읽을 줄 단위를 따로 지정하지 않으면
            linechunk = shape[0]  # 그냥 전체 래스터 행렬의 행 크기로 한꺼번에 읽기, shape는 (196,300)
        chunks = int(math.ceil(shape[0] / float(linechunk)))  # 전체 데이터에서 일정 단위로 읽을 줄 수로 나눈 묶음 수 chunks
        # shape[0]은 196, linechunk는 기본값 1000 그래서 math.ceil(196/1000)하면 1이 나와서 chunks에는 1이 입력됨

        for chunk in range(chunks):  # 이제부터 전체 묶음에서 지정한 줄 단위 수만큼 차례로 처리하기
            # 디버그 출력하기 몇번째 chunk 묶음을 읽고 있는지 프린트해서 알리기
            logger.debug("Writing chunk %d of %d" % (chunk + 1, chunks))
            row = chunk * linechunk  # 몇번째 단위 숫자 묶음인지 알려주는 row

            if row + linechunk > shape[0]:  # 전체 행 숫자를 묶음 chunk로 나누고 나서 나머지가 있을 때의 경우를 처리
                linechunk = shape[0] - row
            # in 1D space -> 왜 1차원 공간인가,내 생각에는 환경변수 1개에 대해서(열 1개에대해서) 처리 하니까 1차원이라는 의미인가?
            start = shape[1] * row
            end = start + shape[1] * linechunk  # start로 읽을 행 위치 지정하고 linechunk만큼 떨어진 행 위치를 end로 지정하기
            # shape[1]은 300, linechunk는 196, end는 58800
            line = target_xs[start:end, :]  # target_xs 전체 데이터 셋 중 부분 dataset을 line으로 지정, line은 (58800,19)
            # 전체 래스터 이미지에서 부분적으로 볼 window 크기 및 window가 있는 위치 지정하기
            window = ((row, row + linechunk), (0, shape[1]))
            # window = ((row_start, row_stop), (col_start, col_stop)) ((0,196),(0,300))

            # Predict
            # 사이킷런의 classifier 인스턴스 clf를 이용하여 target_xs 행렬을 가지고 predict 예측시작
            responses = clf.predict(line)  # line은 (58800,19)이고, responses는 (58800,)
            responses2D = responses.reshape((linechunk, shape[1])).astype("int16")  # responses2D는 (196,300)
            # 모델에서 나온 예측값이 responses에 저장되어 있는데 모두 0이 나오기 때문에 흑백이미지가 저장됨..
            # but 모델에서 나온값인 responses는 무엇을 의미하는가?........
            # rasterio.open -> 일부 데이터만을 가져와서 window크기를 이용해서 도장을 찍어가면서 예측 이미지 생성
            response_ds.write_band(1, responses2D, window=window)
            # 여기서는 responses2D랑 window크기가 같아 도장이 한 번에 찍힘

            if certainty or class_prob:  # 분류 확실성 래스터 생성 옵션이 True 또는 각 클래스별 확률 래스터 생성 옵션이 True이면
                # 부분 데이터셋 line 중에서 classifier 인스턴스 clf로 분류 확실성 래스터를 생성한 proba 데이터셋
                proba = clf.predict_proba(line)  # proba에는 [0.99 0.01]처럼 확률 값이 저장 됨
                # proba.shape는 (58800,2)
            # Certainty
            if certainty:  # 분류 확실성 래스터 생성 옵션이 True면
                certaintymax = proba.max(axis=1)  # certaintymax.shpae는 (58800,) 1차원임
                # 각각의 line 부분 데이터 셋 내에 생성된 proba 값들 중에서 가장 max한 값만 선택하고
                # certaintymax 데이터셋 새로 만들기
                certainty2D = certaintymax.reshape((linechunk, shape[1])).astype("float32")
                # certainymax 데이터셋 행렬 형태를 reshape해서 certainty2D 데이터셋 생성
                # 데이터셋 원소는 float32 타입으로 지정, certainty2D.shape는 (196,300)
                certainty_ds.write_band(1, certainty2D, window=window)
                # rasterio.open 기능 중 write_band를 사용해서 1번째 채널 밴드에서 certainty2D행렬 데이터 셋이
                # window에서 지정해준 subset 내에 있는 래스터 데이터를 centainty_ds에 기록하기
                # (certainty_ds는 분류확실성 래스터의 데이터셋을 기록하도록 write 모드로 전환한 것임)
            # 각 클래스에 대한 확률을 별도의 래스터로 작성
            for i, class_ds in enumerate(class_dss):  # 각 클래스를 분리된 래스터의 확률로 기록하기
                proba_class = proba[:, i]  # 분리된 래스터 확률은 0과 1로 2개임
                # proba[:, 0]에는 래스터의 출현 확률이 기록, proba[:,1]에는 래스터의 비출현 확률이 기록
                # proba_class.shape는 (58800,)로 1차원 구조
                classcert2D = proba_class.reshape((linechunk, shape[1])).astype(  # 1차원구조를 2차원 구조로 바꿔줌
                    "float32")
                class_ds.write_band(1, classcert2D, window=window)

    finally:  # try의 오류 유무에 상관없이 무조건 코드를 실행하는 finally 구간
        response_ds.close()  # rasterio.open으로 래스터 데이터 셋을 읽는 작업 close 하기
        if certainty:  # 분류 확실성 래스터 만드는 옵션인 경우
            certainty_ds.close()  # rasterio.open으로 래스터 데이터 셋을 읽는 작업 close하기
        for class_ds in class_dss:  # 전체 만들어진 클래스 데이터프레임들에 대해서 차례로 read close 하기
            class_ds.close()  # rasterio.open으로 래스터 데이터 셋을 읽는 작업 clsoe하기

def load_targets(explanatory_rasters):
    # explanatory_rasters : 래스터 이미지 파일이 있는 경로들이 있는 리스트 형태
    # ['INPUT\\bclim1.asc', 'INPUT\\bclim10.asc', ...]

    # GDAL(Geospatial Data Abstraction Library) : 래스터 데이터 처리에 많이 사용하는 라이브러리

    """
    Parameters
    ----------
    explanatory_rasters : List of Paths to GDAL rasters containing explanatory variables
    (독립 변수를 포함하는 GDAL 래스터에 대한 경로 목록)
    Returns
    -------
    expl : Array of explanatory variables(독립 변수의 배열로 데이터를 의미)
    raster_info : dict of raster info(래스터 정보가 딕셔너리 형태로 존재)
    """

    explanatory_raster_arrays = []
    transform = None
    shape = None  # 래스터 이미지의 행렬 크기 shape = (행, 열)
    crs = None  # 좌표계 지정 변수 crs

    for raster in explanatory_rasters:  # 래스터 이미지 즉 환경변수 한 개씩 가지고 옴
        # 연산 중에 logger로 raster 변수 작동에 대해 상세한 정보를 열람하기 위해 debug 사용
        # 몇 번째 raster파일(환경변수)이 잘 못 되었는지 출력해줌
        logger.debug(raster)
        # with ... as 구문을 사용하게 되면 파일을 열고 해당 구문이 끝나면 자동으로 닫히게됨
        with rasterio.open(raster) as src:  # rasterio.open 함수를 이용해서 raster 파일을 오픈하기 -> 메모리 절약가능한 코드
            ar = src.read(1)  # TODO band num?
            # 래스터 이미지의 1번째 band의 datasets을 ar로 지정
            # 하나의 환경변수안에서 컬러일 경우 여러 개의 band(layer)가 있음, 흑백은 1개

            # Save or check the geotransform(지리변환 저장 또는 확인)
            if not transform:  # 기하학적 왜곡, 형태 왜곡을 보정하는 affine 변환행렬인 transform 행렬이 None 값으로 있으면
                # 래스터 이미지를 transform함수로 기하학적 왜곡, 형태 왜곡을 보정해줌
                # affine행렬을 선의 평행성을 유지하면서 이미지를 변환해줌, 직사각형 형태로
                transform = src.transform
            else:  # 이미 transform 값이 있으면
                # 저장된 transform 행렬과 실제 래스터 이미지에서 가져온 affine 행렬과 유사한지 다시 검사하기
                assert transform.almost_equals(src.transform)  # 이상 없으면 계속 진행

            # Save or check the shape(저장 및 shape형태 확인)
            if not shape:  # (행, 열) 형태의 데이터가 없으면, 즉 None이면
                shape = ar.shape  # ar의 shape 형태 정보를 shape 변수에 저장
            else:  # 이미 있으면
                assert shape == ar.shape  # 다시 assert로 검사해서 이상 없으면 계속 진행

            # Save or check the geotransform(지리변환 저장 또는 확인)
            # 좌표계 변수 crs가 None이면 -> 좌표계가 디폴트로 지정되어있음.
            if not crs:
                crs = src.crs  # src에서 뽑아온 좌표계 변수를 crs 변수에 저장
            else:  # 이미 있으면
                assert crs == src.crs  # 다시 assert로 검사해서 이상 없으면 계속 진행

        # Flatten in one dimension(1차원으로 Flatten)
        arf = ar.flatten()  # 2차원행렬 형태로 되어있는 래스터 데이터셋을 1차원 행벡터로 flat시키는 flatten()
        # 과장님 주신 코드에 환경변수 1개는 (196,300) 2차원 행렬임
        # 여기서 Flatten하면 환경변수 1개는 196*300 = 58800이 됨
        explanatory_raster_arrays.append(arf)  # explanatory_raster_arrays 리스트에 arf 행벡터를 추가시키기
        # explanatory_raster_arrays 리스트 안에는 각 각의 래스터 데이터셋이 1차원 형태로 들어가서 2중 리스트로 만들어짐

    expl = np.array(explanatory_raster_arrays).T  # 행렬 전치시키기 Transform
    # expl변수 안에는 각각의 래스터 데이터셋의 정보가 열 형태로 19개의 래스터 데이터셋이 저장됨
    # 과장님 코드로 보면 (58800, 19) 여기서 19는 환경변수의 개수임

    raster_info = {
        "transform": transform,  # affine행렬을 -> 왜곡이 사라진 형태로 바쭤줌
        "shape": shape,  # (행, 열)
        "crs": crs,  # 예를들어 좌표계 종류 EPSG:5189 이런 형태
    }  # 딕셔너리 형태로 아핀 행렬 transform, 래스터 이미지 행렬 형태, 좌표계 정보를 저장하기
    return expl, raster_info  # raster_info 딕셔너리 + 래스터 데이터셋 정보들이 담긴 expl 행렬(열벡터 형태)을 반환

def plot_roc_curve(fper, tper,evaluate_path):
    fig = plt.figure(figsize=(5, 2.5))
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()

    # plt.show()
    plt.savefig(evaluate_path)
    plt.cla()


def impute2( 
    target_xs,
    train_y,
    clf,
    raster_info,
    outdir="output",
    linechunk=1000,
    class_prob=True,
    certainty=True,
):

    if not os.path.exists(outdir):
        os.makedirs(outdir) 

    shape = raster_info["shape"] 
		

    profile = {
        "blockxsize": shape[1], 
        "height": shape[0], 
        "blockysize": 1,  
        "count": 1, 
        "crs": raster_info["crs"], 
        "driver": u"GTiff", 
        "dtype": "int16", 
        "nodata": -32768, 
        "tiled": False, 
        "transform": raster_info["transform"], 
        "width": shape[1], 
    } 
    try:
        response_path = os.path.join(outdir, "responses.tif") 
        response_ds = rasterio.open(response_path, "w", **profile) 

        profile["dtype"] = "float32"

        if certainty: 
            certainty_path = os.path.join(outdir, "certainty.tif") 
            certainty_ds = rasterio.open(certainty_path, "w", **profile) 
        class_dss = []
        if class_prob:
            classes = list(np.unique(train_y))
            class_paths = []
            for i, c in enumerate(classes):
                ods = os.path.join(outdir, "probability_%s.tif" % c)
                class_paths.append(ods)
            for p in class_paths:
                class_dss.append(rasterio.open(p, "w", **profile))

      
        if not linechunk: 
            linechunk = shape[0] 
        chunks = int(math.ceil(shape[0] / float(linechunk))) 

        for chunk in range(chunks): 
            logger.debug("Writing chunk %d of %d" % (chunk + 1, chunks)) 
            row = chunk * linechunk 
            if row + linechunk > shape[0]: 
                linechunk = shape[0] - row
            
            start = shape[1] * row
            end = start + shape[1] * linechunk 
            line = target_xs[start:end, :] 

            window = ((row, row + linechunk), (0, shape[1])) 
            responses = clf.predict(line) 
            responses2D = responses.reshape((linechunk, shape[1])).astype("int16") #
            
            response_ds.write_band(1, responses2D, window=window) 


            if certainty or class_prob: 
                proba = clf.score_samples(line) 
            if certainty: 
                certaintymax = minmax_scale(proba)
               
                certainty2D = certaintymax.reshape((linechunk, shape[1])).astype("float32")
								
                certainty_ds.write_band(1, certainty2D, window=window)
								
            
            for i, class_ds in enumerate(class_dss): 
                proba_class = minmax_scale(proba)
                classcert2D = proba_class.reshape((linechunk, shape[1])).astype(
                    "float32"
                )
                class_ds.write_band(1, classcert2D, window=window)

    finally:
        response_ds.close() 
        if certainty: 
            certainty_ds.close() 
        for class_ds in class_dss: 
            class_ds.close()
    
def evaluate_clf(clf, X, y, name, evaluate_path, k=None, test_size=0.5, scoring="roc_auc", feature_names=None):

    X_train, X_test, y_train, y_true = model_selection.train_test_split(X, y, test_size=test_size,
                    shuffle=True, stratify=y) 

    Kfold = ''
    if k:
        kf = model_selection.KFold(n_splits=k) 
        scores = model_selection.cross_val_score(clf, X_train, y_train, cv=kf, scoring=scoring)
        Kfold = name + " %d-fold 교차 검증 정확도: %0.2f (+/- %0.2f)"% (k, scores.mean() * 100, scores.std() * 200)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    AccScore = metrics.accuracy_score(y_true, y_pred)
    clf_report = metrics.classification_report(y_true, y_pred)
    matrix = metrics.confusion_matrix(y_true, y_pred)
    RA_score = metrics.roc_auc_score(y_true, y_pred)

    probs = clf.predict_proba(X_test)
    prob = probs[:, 1]
    fper, tper, thresholds = roc_curve(y_true, prob)
    plot_roc_curve(fper, tper,evaluate_path + '\\_ROC Curve.png')
    if hasattr(clf, "feature_importances_"):
        if feature_names:
            pass
        elif not feature_names:
            feature_names = ["%d" % i for i in range(X.shape[1])]

        fi = clf.feature_importances_
        df_fi = pd.DataFrame({'columns':feature_names, 'importances': fi})
        df_fi = df_fi[df_fi['importances'] > 0]
        df_fi = df_fi.sort_values(by=['importances'], ascending=False).head(5)

        fig = plt.figure(figsize=(8,4.5))
        ax = sns.barplot(x=df_fi['columns'], y=df_fi['importances'])
        ax.set_xticklabels(df_fi['columns'], rotation=80, fontsize=13)
        plt.tight_layout()
        plt.savefig(evaluate_path+'\\_Feature importances.png')
        plt.cla() # plt 초기화
        return Kfold, AccScore, clf_report, matrix, RA_score, fi
    return Kfold, AccScore, clf_report, matrix, RA_score
