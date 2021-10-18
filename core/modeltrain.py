import pandas as pd
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from pathlib import Path
from core import modellists
import json
import datetime
# matplotlib.use('Agg')



def getXandY(data, xcol, ycol) :
    """
    X와 Y데이터 나누기
    """
    df = pd.read_json(data, orient='split')

    X = df[xcol]
    Y = df[[ycol]]

    X_js = X.to_json(orient='split')
    Y_js = Y.to_json(orient='split')

    return X_js, Y_js


def getScaler(xdata, scaler) :
    """
    데이터 scaling
    scalerlist : ['적용 안 함', 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'Normalizer']
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
    df = pd.read_json(xdata, orient='split')
    cols = df.columns.tolist()

    if scaler == 'StandardScaler' :
        scale = StandardScaler().fit_transform(df)
    elif scaler == 'RobustScaler' :
        scale = RobustScaler().fit_transform(df)
    elif scaler == 'MinMaxScaler' :
        scale = MinMaxScaler().fit_transform(df)
    elif scaler == 'Normalizer' :
        scale = Normalizer().fit_transform(df)
    else :
        df_js = df.to_json(orient='split')
        return df_js

    df = pd.DataFrame(scale, columns=cols)
    df_js = df.to_json(orient='split')

    return df_js


def splitData(xdata, ydata, test, valid) :
    """
    데이터 split
    """
    from sklearn.model_selection import train_test_split

    xdf = pd.read_json(xdata, orient='split')
    ydf = pd.read_json(ydata, orient='split')

    X1, X_test, Y1, Y_test = train_test_split(xdf, ydf, test_size=test, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X1, Y1, test_size=valid, random_state=42)

    X_train_js = X_train.to_json(orient='split')
    Y_train_js = Y_train.to_json(orient='split')
    X_test_js = X_test.to_json(orient='split')
    Y_test_js = Y_test.to_json(orient='split')
    X_valid_js = X_valid.to_json(orient='split')
    Y_valid_js = Y_valid.to_json(orient='split')

    return X_train_js, Y_train_js, X_test_js, Y_test_js, X_valid_js, Y_valid_js


def getAmount(data) :
    """
    데이터갯수 구하기
    """
    df = pd.read_json(data, orient='split')
    amount = len(df.index.tolist())

    return amount


def getFilename(model, pname, dname, mid):
    """
    모델의 pkl파일을 저장하고 파일 경로를 텍스트로 리턴받는다
    """
    import pickle

    dir = Path(__file__).resolve().parent.parent
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    pklfile = pickle.dump(model, open(f'{dir}/media/models/{pname}_{dname}_model_{mid}_{now}.pkl', 'wb'))
    mdl_pkl = f'{dir}/media/models/{pname}_{dname}_model_{mid}_{now}.pkl'

    return mdl_pkl


def getRegGraph(Y_va, pred) :
    """
    회귀모델 산점도+추세선 그래프용 json 리턴
    """
    # predict와 정답컬럼 비교 df
    Y_va = Y_va.reset_index(drop=True)
    preddf = pd.DataFrame(pred.tolist(), columns=['predict'])
    validdf = pd.concat([Y_va, preddf], axis=1)
    validdf.columns = ['Answer', 'Predict']
    graph = validdf.to_json(orient='split')

    return graph


def getRegImportance(model, xcols) :
    """
    회귀모델 변수중요도 json 리턴
    """
    importance = model.coef_
    coef = pd.DataFrame(importance, columns=xcols).T
    coef.columns = ['Importance']
    coef = coef.sort_values(by='Importance', ascending=False)
    pl_im = coef.to_json(orient='split')

    try :
        best_feat = coef.index[:2].tolist()
        return pl_im, best_feat
    except :
        best_feat = coef.index[:1].tolist()
        return pl_im, best_feat


def getRMSEMAE(Y_va, pred) :
    """
    회귀모델을 위한 RMSE, MSE, RMSPE, MAE, MAPE 딕셔너리 리턴
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

    # 검증셋 RMSE-MAE 표 추가 (딕셔너리 json)
    Y_va = Y_va.values
    rmse = mean_squared_error(Y_va, pred, squared=False)
    mse = mean_squared_error(Y_va, pred, squared=True)
    epsilon = 1e-10
    rmspe = (np.sqrt(np.mean(np.square((Y_va - pred) / (Y_va + epsilon))))) * 100
    mae = mean_absolute_error(Y_va, pred)
    mape = mean_absolute_percentage_error(Y_va, pred)
    rmse_mae = {"RMSE": rmse,
                "RMSPE": rmspe,
                "MSE": mse,
                "MAE": mae,
                "MAPE": mape, }

    return rmse_mae


# DTREE 플롯
def getDTreePlot(model, X, Y, xcols, ycols, rorc, class_names, pname, dname, mid):
    """
    랜덤포레스트의 Dtreeviz 그래프를 리턴

    """
    from dtreeviz.trees import dtreeviz
    dir = Path(__file__).resolve().parent.parent
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file = f'{dir}/media/graph/{pname}_{dname}_model_{mid}_{now}.svg'

    # plt.figure(dpi=600)
    if rorc == 'r' :
        viz = dtreeviz(model.estimators_[0], X.values, Y.values.reshape(-1), target_name=ycols[0], feature_names=xcols, scale=2.0)
    else :
        viz = dtreeviz(model.estimators_[0], X.values, Y.values.reshape(-1), target_name=ycols[0], feature_names=xcols, class_names=class_names, fancy=True, scale=2.0)

    viz.save(file)

    return file


# DTREE 플롯
def getXGBDTreePlot(model, X, Y, xcols, ycols, rorc, class_names, pname, dname, mid):
    """
    XGBoost의 Dtreeviz 그래프를 리턴

    """
    from dtreeviz.trees import dtreeviz
    dir = Path(__file__).resolve().parent.parent
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file = f'{dir}/media/graph/{pname}_{dname}_model_{mid}_{now}.svg'

    # plt.figure(dpi=600)
    if rorc == 'r' :
        viz = dtreeviz(model, X.values, Y.values.reshape(-1), target_name=ycols[0], feature_names=xcols, tree_index=0, scale=2.0)
    else :
        viz = dtreeviz(model, X.values, Y.values.reshape(-1), target_name=ycols[0], feature_names=xcols, class_names=class_names, tree_index=1, fancy=True, scale=2.0)

    viz.save(file)

    return file


# 기존플롯
def getTreePlot(model, xcols, pname, dname, mid):
    """
    랜덤포레스트 회귀/분류의 기본 Treeplot 이미지 경로 리턴
    """
    from sklearn import tree
    dir = Path(__file__).resolve().parent.parent
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file = f'{dir}/media/graph/{pname}_{dname}_model_{mid}_{now}.png'

    plt.figure(dpi=600)
    tree.plot_tree(model.estimators_[0],
                   feature_names = xcols,
                   filled = True)
    plt.savefig(file)
    plt.close()

    return file


# 기존플롯
def getXGBTreePlot(model, pname, dname, mid):
    """
    XGBoost 회귀/분류의 기본 TreePlot 이미지 경로 리턴
    **Graphviz 오류를 해결하기 위해 modellist를 통해 graphviz 경로가 추가되어있는지 매번 확인(없으면 자동추가)
    """
    from xgboost import plot_tree
    mdl = modellists.GetModel()

    dir = Path(__file__).resolve().parent.parent
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file = f'{dir}/media/graph/{pname}_{dname}_model_{mid}_{now}.png'

    plot_tree(model)
    plt.savefig(file)
    plt.close()

    return file


def getTreeImportance(model, xcols) :
    """
    Tree모델의 Feature Importance json 리턴
    """
    fi = pd.DataFrame(model.feature_importances_, index=xcols)
    fi_json = fi.to_json(orient='split')
    try :
        best_feat = fi.index[:2].tolist()
        return fi_json, best_feat
    except :
        best_feat = fi.index[:1].tolist()
        return fi_json, best_feat


def getConfusion(Y_va, pred) :
    """
    분류용 Confusion Matrix를 위한 json 리턴
    """
    from sklearn.metrics import confusion_matrix

    cf = confusion_matrix(Y_va, pred)
    confusion = pd.DataFrame(columns=['NG', 'OK'], index=['NG', 'OK'])
    confusion.loc['NG', 'NG'] = cf[0][0]
    confusion.loc['NG', 'OK'] = cf[0][1]
    confusion.loc['OK', 'NG'] = cf[1][0]
    confusion.loc['OK', 'OK'] = cf[1][1]

    confusion_json = confusion.to_json(orient='split')

    return confusion_json


def getROC(Y_va, proba) :
    """
    분류용 ROC Curve를 위한 json 리턴
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    fprs, tprs, _ = roc_curve(Y_va, proba)

    roc_curve = pd.DataFrame(columns=['FPR', 'TPR'])
    roc_curve['FPR'] = fprs
    roc_curve['TPR'] = tprs

    roc_curve_json = roc_curve.to_json(orient='split')

    return roc_curve_json


def getLearningCurve(model) :
    """
    MLP를 위한 학습곡선(손실곡선) dict 리턴
    """
    loss = model.loss_curve_
    loss_dict = {"Loss Curve" : loss}
    return loss_dict



# sklearn PDP -> df로 만드는 과정에서 오류 발생
def getskPDP(model, X_va, best_feat) :
    from sklearn.inspection import partial_dependence

    pdps = partial_dependence(model, X_va, best_feat)[0][0]
    # pdps = []
    # for b in best_feat :
    #     pdps.append(partial_dependence(model, X_va, b)[0][0].tolist())
    # print(pdps)
    pdps_df = pd.DataFrame(pdps, index=best_feat).T
    pdp_json = pdps_df.to_json(orient='split')

    return pdp_json
    return None


# PDPbox PDP
def getPDP(model, X_va, xcols, best_feat) :
    from pdpbox.pdp import pdp_interact, pdp_isolate
    import copy

    if len(best_feat) == 2 :
        pdp_interact_out = pdp_interact(model, X_va, xcols, best_feat)

        # sauce from pdpbox library
        pdp_mx_temp = copy.deepcopy(pdp_interact_out.pdp)
        for feature, feature_type, mark in zip(pdp_interact_out.features, pdp_interact_out.feature_types, ['x', 'y']):
            if feature_type in ['numeric', 'binary']:
                pdp_mx_temp[mark] = pdp_mx_temp[feature]
            else:
                # for onehot encoding feature, need to map to numeric representation
                pdp_mx_temp[mark] = pdp_mx_temp[feature].apply(lambda x: list(x).index(1), axis=1)
        pdp_mx_temp = pdp_mx_temp[['x', 'y', 'preds']].sort_values(by=['x', 'y'], ascending=True)

        pdp_inter = copy.deepcopy(pdp_mx_temp['preds'].values)
        n_grids_x, n_grids_y = len(pdp_interact_out.feature_grids[0]), len(pdp_interact_out.feature_grids[1])
        pdp_mx = pdp_inter.reshape((n_grids_x, n_grids_y)).T
        X, Y = np.meshgrid(pdp_interact_out.feature_grids[0], pdp_interact_out.feature_grids[1])

        pdp_df = pd.DataFrame(pdp_mx, columns=pdp_interact_out.pdp_isolate_outs[0].display_columns,
                              index=pdp_interact_out.pdp_isolate_outs[1].display_columns)

        pdp_json = pdp_df.to_json(orient='split')
        pdp_re = {"column_count" : 2,
                  "column_names" : best_feat,
                  "pdp_json" : json.loads(pdp_json)}

    if len(best_feat) == 1 :
        pdp_isolate_out = pdp_isolate(model, X_va, xcols, best_feat[0])

        # sauce from pdpbox library
        feature_type = pdp_isolate_out.feature_type
        feature_grids = pdp_isolate_out.feature_grids
        display_columns = pdp_isolate_out.display_columns
        percentile_info = pdp_isolate_out.percentile_info
        percentile_xticklabels = list(percentile_info)
        if feature_type == 'binary' or feature_type == 'onehot' :
            x = range(len(feature_grids))
            xticks = x
            xticklabels = list(display_columns)
        else:
            x = feature_grids
        ice_lines = copy.deepcopy(pdp_isolate_out.ice_lines)
        pdp_y = copy.deepcopy(pdp_isolate_out.pdp)

        std = ice_lines[feature_grids].std().values

        upper = pdp_y + std
        lower = pdp_y - std
        all = [pdp_y, upper, lower]
        pdp_df = pd.DataFrame(all, index=['y', 'upper_y', 'lower_y'],
                              columns=xticklabels)

        pdp_json = pdp_df.to_json(orient='split')
        pdp_re = {"column_count" : 1,
                  "column_names" : best_feat[0],
                  "pdp_json" : json.loads(pdp_json)}

    return pdp_re


def getLimeImportance(model, X, X_va, xcols, ycols, rorc) :
    from lime.lime_tabular import LimeTabularExplainer
    from lime import submodular_pick

    if rorc == 'r' :
        explainer = LimeTabularExplainer(X, mode='regression', feature_names=xcols, class_names=ycols,
                                         discretize_continuous=False)
        # i = np.random.randint(0, X_va.shape[0])
        # exp = explainer.explain_instance(X_va.values[i], model.predict, top_labels=1)
        # exp_list = list(map(list, exp.as_list()))
        # cols = [x[0] for x in exp_list]
        # vals = [x[1] for x in exp_list]
        # lime_df = pd.DataFrame(vals, index=cols, columns=['Importance'])
        exp = submodular_pick.SubmodularPick(explainer, X.values, model.predict, sample_size=20)

    elif rorc == 'c' :
        explainer = LimeTabularExplainer(X, mode='classification', feature_names=xcols, class_names=ycols,
                                         discretize_continuous=False)
        exp = submodular_pick.SubmodularPick(explainer, X.values, model.predict_proba, sample_size=20)
    list_exp = []
    for e in exp.explanations :
        dict_exp = {}
        for i in list(e.as_map().values())[0] :
            dict_exp[xcols[i[0]]] = i[1]
        list_exp.append(dict_exp)

    df1 = pd.DataFrame(list_exp)
    # df1 = pd.DataFrame([dict(this.as_map().values()) for this in exp.explanations])
    vals = df1.mean().values.tolist()
    idx = df1.mean().index.tolist()
    abvals = list(map(abs, vals))
    lime_df = pd.DataFrame([vals, abvals]).T
    lime_df.columns = ['Importance', 'ABS']
    lime_df.index = idx

    # for i in exp.explanations:
    #     print(dir(i))
    #     print(i.as_list())
    #
    # exp_list = [list(map(list, x.as_list())) for x in exp.explanations]
    # cols = [x[0] for x in exp_list]
    # vals = [x[1] for x in exp_list]
    # abvals = list(map(abs, vals))
    # lime_df = pd.DataFrame([vals, abvals]).T
    # lime_df.columns = ['Importance', 'ABS']
    # lime_df.index = cols

    lime_df.sort_values('ABS', ascending=False, inplace=True)
    lime_df.drop(['ABS'], axis=1, inplace=True)
    lime_js = lime_df.to_json(orient='split')
    # print(lime_js)
    # print(lime_df.index)

    best_feat = lime_df.index[:2].tolist()

    return lime_js, best_feat

# # TODO : 너무 느려서 사용보류
# def getSHAPImportance(model, X, X_va, xcols, ycols, rorc) :
#     import shap
#
#     # explainer = shap.KernelExplainer(model.predict, X)
#     # shap_values = explainer.shap_values(X_va)
#     # shap.summary_plot(shap_values, X_va, feature_names=xcols)
#     if rorc == 'r' :
#         explainer = shap.KernelExplainer(model.predict, X)
#         shap_values = explainer.shap_values(X_va, nsamples=100)
#     elif rorc == 'c' :
#         explainer = shap.KernelExplainer(model.predict_proba, X)
#         shap_values = explainer.shap_values(X_va, nsamples=100)
#     # shap_values = explainer(X[:100])
#
#     print(shap_values)
#
#     pass



# TODO : MLP 기본설정 연구원님들 의견 반영할 것

def getModel(x, y, x_va, y_va, mname, params, pname, dname, mid) :
    """
    모델 적용해 pkl생성 후 결과정보 리턴

    회귀모델 : '선형 회귀', '랜덤포레스트 회귀', 'XGBoost 회귀', 'MLP 회귀'
    분류모델 : '로지스틱 회귀', '랜덤포레스트 분류', 'XGBoost 분류', 'MLP 분류'

    Tree params : 'n_estimators'(트리의 개수), 'max_depth'(깊이) -> str으로 되어있음!

    """

    X = pd.read_json(x, orient='split')
    Y = pd.read_json(y, orient='split')
    X_va = pd.read_json(x_va, orient='split')
    Y_va = pd.read_json(y_va, orient='split')
    xcols = X.columns.tolist()
    ycols = Y.columns.tolist()


    if mname == '선형 회귀' :
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, Y)

        pred = model.predict(X_va)

        # PKL 저장 & 모델경로 추가
        mdl_pkl = getFilename(model, pname, dname, mid)

        # 검증셋 산점도+추세선 추가 (json)
        graph = getRegGraph(Y_va, pred)

        # 선형회귀 변수중요도 추가 (json)
        pl_im, best_feat = getRegImportance(model, xcols)

        # 검증셋 RMSE-MAE 표 추가 (딕셔너리)
        rmse_mae = getRMSEMAE(Y_va, pred)

        # PDP
        pdp = getPDP(model, X_va, xcols, best_feat)

        return mdl_pkl, pl_im, graph, rmse_mae, pdp


    elif mname == '랜덤포레스트 회귀' :
        from sklearn.ensemble import RandomForestRegressor

        n_es = int(params.split(', ')[0].split(' = ')[1])
        max_dep = int(params.split(', ')[1].split(' = ')[1])

        model = RandomForestRegressor(n_estimators=n_es, max_depth=max_dep).fit(X, Y)
        pred = model.predict(X_va)

        # PKL 저장 & 모델경로 추가
        mdl_pkl = getFilename(model, pname, dname, mid)

        # 변수중요도 추가(json)
        pl_im, best_feat = getTreeImportance(model, xcols)

        # 검증셋 산점도+추세선 추가 (json)
        graph = getRegGraph(Y_va, pred)

        # 검증셋 RMSE-MAE 표 추가 (딕셔너리)
        rmse_mae = getRMSEMAE(Y_va, pred)

        # PDP
        pdp = getPDP(model, X_va, xcols, best_feat)

        # TreePlot 추가(이미지)
        # treeplot = getTreePlot(model, xcols, pname, dname, mid)
        rorc = 'r'
        class_names = []
        treeplot = getDTreePlot(model, X, Y, xcols, ycols, rorc, class_names, pname, dname, mid)

        return mdl_pkl, pl_im, treeplot, graph, rmse_mae, pdp


    elif mname == 'XGBoost 회귀' :
        from xgboost import XGBRegressor

        n_es = int(params.split(', ')[0].split(' = ')[1])
        max_dep = int(params.split(', ')[1].split(' = ')[1])

        model = XGBRegressor(n_estimators=n_es, max_depth=max_dep).fit(X, Y)
        pred = model.predict(X_va)

        # PKL 저장 & 모델경로 추가
        mdl_pkl = getFilename(model, pname, dname, mid)

        # 변수중요도 추가(딕셔너리 형태)
        pl_im, best_feat = getTreeImportance(model, xcols)

        # 검증셋 산점도+추세선 추가 (딕셔너리 json)
        graph = getRegGraph(Y_va, pred)

        # 검증셋 RMSE-MAE 표 추가 (딕셔너리 json)
        rmse_mae = getRMSEMAE(Y_va, pred)

        # PDP
        pdp = getPDP(model, X_va, xcols, best_feat)

        # TreePlot 추가(이미지)
        rorc = 'r'
        class_names = []
        treeplot = getXGBDTreePlot(model, X, Y, xcols, ycols, rorc, class_names, pname, dname, mid)

        return mdl_pkl, pl_im, treeplot, graph, rmse_mae, pdp


    elif mname == 'MLP 회귀' :
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08).fit(X, Y)
        pred = model.predict(X_va)

        # PKL 저장 & 모델경로 추가
        mdl_pkl = getFilename(model, pname, dname, mid)

        # 학습곡선(손실곡선) 추가 (딕셔너리)
        losscurve = getLearningCurve(model)

        # 검증셋 산점도+추세선 추가 (json)
        graph = getRegGraph(Y_va, pred)

        # 검증셋 RMSE-MAE 표 추가 (딕셔너리)
        rmse_mae = getRMSEMAE(Y_va, pred)

        # 변수중요도(리턴내용에 추가해야할지는 고민해볼 것)
        rorc = 'r'
        pl_im,  best_feat = getLimeImportance(model, X, X_va, xcols, ycols, rorc)

        # PDP
        pdp = getPDP(model, X_va, xcols, best_feat)

        return mdl_pkl, losscurve, graph, rmse_mae, pdp


    elif mname == '로지스틱 회귀' :
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit(X, Y)
        pred = model.predict(X_va)
        proba = model.predict_proba(X_va)[:, 1]

        # PKL 저장 & 모델경로 추가
        mdl_pkl = getFilename(model, pname, dname, mid)

        # 선형회귀 변수중요도 추가 (딕셔너리 json)
        pl_im, best_feat = getRegImportance(model, xcols)

        # Confusion Matrix 추가
        confusion = getConfusion(Y_va, pred)

        # ROC 추가
        roc = getROC(Y_va, proba)

        # PDP
        pdp = getPDP(model, X_va, xcols, best_feat)

        return mdl_pkl, pl_im, confusion, roc, pdp


    elif mname == '랜덤포레스트 분류' :
        from sklearn.ensemble import RandomForestClassifier

        n_es = int(params.split(', ')[0].split(' = ')[1])
        max_dep = int(params.split(', ')[1].split(' = ')[1])

        model = RandomForestClassifier(n_estimators=n_es, max_depth=max_dep).fit(X, Y)
        pred = model.predict(X_va)
        proba = model.predict_proba(X_va)[:, 1]

        # PKL 저장 & 모델경로 추가
        mdl_pkl = getFilename(model, pname, dname, mid)


        # 변수중요도 추가(딕셔너리 형태)
        pl_im, best_feat = getTreeImportance(model, xcols)

        # Confusion Matrix 추가
        confusion = getConfusion(Y_va, pred)

        # ROC 추가
        roc = getROC(Y_va, proba)

        # PDP
        pdp = getPDP(model, X_va, xcols, best_feat)

        # TreePlot 추가(이미지)
        # treeplot = getTreePlot(model, xcols, pname, dname, mid)
        rorc = 'c'
        class_names = Y.iloc[:, 0].unique().tolist()
        treeplot = getDTreePlot(model, X, Y, xcols, ycols, rorc, class_names, pname, dname, mid)

        return mdl_pkl, pl_im, treeplot, confusion, roc, pdp


    elif mname == 'XGBoost 분류' :
        from xgboost import XGBClassifier

        n_es = int(params.split(', ')[0].split(' = ')[1])
        max_dep = int(params.split(', ')[1].split(' = ')[1])

        model = XGBClassifier(n_estimators=n_es, max_depth=max_dep).fit(X, Y)
        pred = model.predict(X_va)
        proba = model.predict_proba(X_va)[:, 1]

        # PKL 저장 & 모델경로 추가
        mdl_pkl = getFilename(model, pname, dname, mid)


        # 변수중요도 추가(딕셔너리 형태)
        pl_im, best_feat = getTreeImportance(model, xcols)

        # Confusion Matrix 추가
        confusion = getConfusion(Y_va, pred)

        # ROC 추가
        roc = getROC(Y_va, proba)

        # PDP
        pdp = getPDP(model, X_va, xcols, best_feat)

        # TreePlot 추가(이미지)
        # treeplot = getXGBTreePlot(model, pname, dname, mid)
        rorc = 'c'
        class_names = Y.iloc[:,0].unique().tolist()
        treeplot = getXGBDTreePlot(model, X, Y, xcols, ycols, rorc, class_names, pname, dname, mid)

        return mdl_pkl, pl_im, treeplot, confusion, roc, pdp


    elif mname == 'MLP 분류' :
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic',
                              solver='sgd', alpha=0.01, batch_size=32,
                              learning_rate_init=0.1, max_iter=500).fit(X, Y)

        pred = model.predict(X_va)
        proba = model.predict_proba(X_va)[:, 1]

        # PKL 저장 & 모델경로 추가
        mdl_pkl = getFilename(model, pname, dname, mid)

        # 학습곡선(손실곡선) 추가 (딕셔너리)
        losscurve = getLearningCurve(model)

        # Confusion Matrix 추가
        confusion = getConfusion(Y_va, pred)

        # ROC 추가
        roc = getROC(Y_va, proba)

        # 변수중요도(리턴내용에 추가해야할지는 고민해볼 것)
        rorc = 'c'
        pl_im, best_feat = getLimeImportance(model, X, X_va, xcols, ycols, rorc)

        # PDP
        pdp = getPDP(model, X_va, xcols, best_feat)

        return mdl_pkl, losscurve, confusion, roc, pdp

