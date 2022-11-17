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
import pandas

def speed_to_eta(pred_data, BASEDIR, city):
    fn = BASEDIR / "road_graph" / city / "road_graph_edges.parquet"
    df_egde = pandas.read_parquet(fn)

    pred_data['vol'] = np.argmax(pred_data[["pred_vol_1","pred_vol_3","pred_vol_5"]].values,axis=1)
    y = []
    for i in range(100):
        y.append(df_egde[['u','v','flow','length_meters']])
    y = pandas.concat(y)
    pred_data[['u','v','flow','length_meters']] = y
    del y, df_egde
    pred_data['eta'] = pred_data['length_meters']/pred_data['pred_speed']*3.6
    pred_data = pred_data[["test_idx","u","v","eta"]]
    pred_data["u"] = pred_data["u"].apply(lambda x:int(x))
    pred_data["v"] = pred_data["v"].apply(lambda x:int(x))

    eta_results = []
    for i in range(100):
        test_df = pred_data[pred_data['test_idx']==i]
        edge_eta_map = {}
        
        for u, v, e in zip(test_df["u"], test_df["v"], test_df["eta"]):
            edge_eta_map[(u, v)] = e
        rgss_df = pandas.read_parquet(BASEDIR / 'road_graph' / city / 'road_graph_supersegments.parquet')
        for identifier, nodes in zip(rgss_df["identifier"], rgss_df["nodes"]):
            path_eta = 0
            for n1, n2 in zip(nodes[:-1], nodes[1:]):
                e = (n1, n2)
                edge_eta = edge_eta_map[e]
                if edge_eta > 1800:
                    edge_eta = min(2400,edge_eta)
                path_eta += edge_eta
            path_eta = min(3600, path_eta)
            eta_result = {
                        "test_idx" :i,
                        "identifier": identifier,
                        "eta": path_eta
                    }
            eta_results.append(eta_result)
    
    eta_df = pandas.DataFrame(eta_results)
    del pred_data, eta_results
    assert eta_df["eta"].max() <= 3600.0
    return eta_df

BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)
submission_name_input = "../data/submissions/GNN_result_eta/" 
submission_name_ouput = "GNNv10_merge_all_eta"
cities = ["london","melbourne","madrid"]
(BASEDIR / "submissions" / submission_name_ouput ).mkdir(exist_ok=True, parents=True)

for city in cities:

    (BASEDIR / "submissions" / submission_name_ouput / city / "labels").mkdir(exist_ok=True, parents=True)
    inputs_files = sorted(glob(submission_name_input + city +"/labels/eta_speed_test_*.parquet"))
    print(inputs_files)
    merge_num = 0

    df_city = load_df_from_parquet(inputs_files[0])
    pred_city_1 = df_city[["pred_speed", "pred_vol_1", "pred_vol_3","pred_vol_5"]].values
    pred_city = np.zeros_like(pred_city_1)
    for i in range(len(inputs_files)):
        df_city1 = load_df_from_parquet(inputs_files[i])
        pred_city1 = df_city1[["pred_speed", "pred_vol_1", "pred_vol_3","pred_vol_5"]].values
        pred_city = pred_city + pred_city1
        merge_num += 1
        
    print(city,"merge:",merge_num)
    pred_city = pred_city/merge_num
    df_city[["pred_speed", "pred_vol_1", "pred_vol_3","pred_vol_5"]] = pred_city
    eta_df = speed_to_eta(df_city, BASEDIR, city)
    eta_df.to_parquet(BASEDIR / "submissions" / submission_name_ouput / city / "labels" / f"eta_labels_test.parquet", compression='snappy')
    
submission_zip = BASEDIR / "submissions" / f"{submission_name_ouput}.zip"
with zipfile.ZipFile(submission_zip, "w") as z:
    for city in cities:
        z.write(
            filename=BASEDIR / "submissions" / submission_name_ouput / city / "labels" / f"eta_labels_test.parquet",
            arcname=os.path.join(city, "labels", f"eta_labels_test.parquet"),
        )

