import os
import sys
sys.path.insert(0, os.path.abspath("../"))
from pathlib import Path
import numpy as np
import pandas as pd
import t4c22
from t4c22.t4c22_config import load_basedir
from glob import glob
from t4c22.misc.parquet_helpers import load_df_from_parquet
from t4c22.misc.parquet_helpers import write_df_to_parquet
import zipfile


BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)
submission_name_input = "../data/submission/GNN_result_cc/"
submission_name_ouput = "ensemble_cc_result"

cities = ["london","melbourne","madrid"]
(BASEDIR / "submission" / submission_name_ouput ).mkdir(exist_ok=True, parents=True)


for city in cities:
    
    (BASEDIR / "submission" / submission_name_ouput / city / "labels").mkdir(exist_ok=True, parents=True)
    
    inputs_files = sorted(glob(submission_name_input + city +"/labels/*"))
   
    merge_num = 0

    df_city = load_df_from_parquet(inputs_files[0])
    pred_city_1 = df_city[["logit_green", "logit_yellow", "logit_red"]].values
    pred_city = np.zeros_like(pred_city_1)
    for i in range(len(inputs_files)):
        df_city1 = load_df_from_parquet(inputs_files[i])
        pred_city1 = df_city1[["logit_green", "logit_yellow", "logit_red"]].values
        pred_city = pred_city + pred_city1
        merge_num += 1

    print(city,"merge:",merge_num)
    pred_city = pred_city/merge_num
    df_city[["logit_green", "logit_yellow", "logit_red"]] = pred_city
    write_df_to_parquet(df=df_city, fn=BASEDIR / "submission" / submission_name_ouput / city / "labels" / f"cc_labels_test.parquet")

submission_zip = BASEDIR / "submission" / f"{submission_name_ouput}.zip"
with zipfile.ZipFile(submission_zip, "w") as z:
    for city in cities:
        z.write(
            filename=BASEDIR / "submission" / submission_name_ouput / city / "labels" / f"cc_labels_test.parquet",
            arcname=os.path.join(city, "labels", f"cc_labels_test.parquet"),
        )
