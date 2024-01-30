import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from tqdm import tqdm

# pip install pysodmetrics
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

method='pvt'
for _data_name in['DUT-OMRON', 'DUTS-TE', 'ECSSD','HKU-IS', 'PASCAL-S', 'SOD']:
    mask_root = './Data/SOD/{}/GT'.format(_data_name)
    pred_root = './test_result/{}/{}/pred_1/'.format(method, _data_name)
    mask_name_list = sorted(os.listdir(mask_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "MAE": mae,
        "meanEm": em["curve"].mean()
    }

    print(results)
    file=open("./v1_pred_1_results.txt", "a")
    file.write(method+' '+_data_name+' '+str(results)+'\n')