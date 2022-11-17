#  Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import sys
sys.path.insert(0, os.path.abspath("../"))
import statistics
from collections import defaultdict
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
import tqdm
from IPython.core.display import HTML
from IPython.display import display
from torch import nn
from torch_geometric.nn import MessagePassing,TransformerConv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import t4c22
from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config
from t4c22.t4c22_config import class_fractions
from t4c22.t4c22_config import load_basedir
from t4c22.dataloading.t4c22_dataset import T4c22Dataset
from t4c22.plotting.plot_congestion_classification import plot_segment_classifications_simple
from t4c22.misc.notebook_helpers import restartkernel  # noqa:F401
import torch
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from t4c22.misc.parquet_helpers import write_df_to_parquet
import zipfile
from glob import glob
from model.utils import to_var, weight_init

class GNN_Layer(MessagePassing):
    """
    Parameters
    ----------
    in_features : int
        Dimensionality of input features.
    out_features : int
        Dimensionality of output features.
    """

    def __init__(self, in_features, out_features, hidden_features):
        super(GNN_Layer, self).__init__(node_dim=-2, aggr="mean")

        self.message_net = nn.Sequential(
            nn.Linear(2 * in_features, hidden_features), Swish(), nn.BatchNorm1d(hidden_features), nn.Linear(hidden_features, out_features), Swish()
        )
        self.update_net = nn.Sequential(nn.Linear(in_features + hidden_features, hidden_features), Swish(), nn.Linear(hidden_features, out_features), Swish())

    def forward(self, x, edge_index):
        """Propagate messages along edges."""
        x = self.propagate(edge_index, x=x)
        # x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j):
        """Message update."""
        message = self.message_net(torch.cat((x_i, x_j), dim=-1))
        return message

    def update(self, message, x):
        """Node update."""
        x += self.update_net(torch.cat((x, message), dim=-1))
        return x
class Edge_Attr(nn.Module):
    attr_dims = [ ("importance", 0, 8, 5), ("oneway", 1, 2, 2),('tunnel', 2, 2, 2),('lanes', 3, 6, 3)]
    def __init__(self,num_edge_dim):
        super(Edge_Attr, self).__init__()
        for name, i , dim_in, dim_out in Edge_Attr.attr_dims:
            self.add_module("attr-" + name, nn.Embedding(dim_in, dim_out))
        self.process_coords = torch.nn.Linear(20+1, 64)

    def forward(self, num_attr, cc_attr,y_init):
        em_list = []
        for name, i, dim_in, dim_out in Edge_Attr.attr_dims:
            embed = getattr(self, "attr-" + name)
            attr_t = cc_attr[:,:,i]
            attr_t = embed(attr_t)
            em_list.append(attr_t)
        em_list.append(num_attr)
        em_list.append(y_init)
        out = torch.cat(em_list, -1)
        out = nn.LeakyReLU()(self.process_coords(out))
        return out

    def out_size(self):
        sz = 0
        for name, dim_in, dim_out in Attr.attr_dims:
            sz += dim_out

class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_edge_dim, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(torch.nn.Linear(hidden_dim, 3))
        self.lins1 = torch.nn.ModuleList()
        self.lins1.append(torch.nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.lins1.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.lins1.append(torch.nn.Linear(hidden_dim, 1))

        self.lins2 = torch.nn.ModuleList()
        self.lins2.append(torch.nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.lins2.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.lins2.append(torch.nn.Linear(hidden_dim, 3))
        self.swish = Swish()

        self.dropout = dropout
        self.__drop_p = dropout
        self.__activation = ReLU()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self,x):

        x1 = x
        x2 = x
        for lin in self.lins[:-1]:
            x = lin(x)
            res = nn.LeakyReLU()(x)
            x = x + res
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        
        for lin in self.lins1[:-1]:
            x1 = lin(x1)
            res = nn.LeakyReLU()(x1)
            x1 = x1 + res
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        x1 = self.lins1[-1](x1)

        for lin in self.lins2[:-1]:
            x2 = lin(x2)
            res = nn.LeakyReLU()(x2)
            x2 = x2 + res
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.lins2[-1](x2)

        return x,x1,x2

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(in_features = city_attr["counters"], out_features =city_attr["counters"], bias = True)
        self.hidden2 = nn.Linear(city_attr["counters"],city_attr["counters"])
        self.predict = nn.Linear(city_attr["counters"],city_attr["edges"])
        self.out = nn.Linear(1,3)
    
    def forward(self,x):

        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        output = self.predict(x)
        output = self.out(output.unsqueeze(-1))
        return output
dropout = 0.05

class TrfEdgeNet(torch.nn.Module):
    def __init__(self,  num_node_dim, 
                        num_edge_dim, 
                        num_edge_classes,
                        trf_num_layers,
                        mlp_num_layers,
                        trf_hidden_dim,
                        hidden_dim=32,  
                        drop_p=0.05, 
                        heads=6,
                        concat=False, 
                        aggr="add"):
        super(TrfEdgeNet, self).__init__()
        self.trf_num_layers = trf_num_layers
       
        self.cgnn = torch.nn.ModuleList(modules=[GNN_Layer(hidden_dim//2, hidden_dim//2, hidden_dim//2) for _ in range(trf_num_layers)])
        self.__activation = nn.LeakyReLU()
        self.__drop_p = drop_p
        self.predictor = LinkPredictor(hidden_dim, hidden_dim, num_edge_classes, mlp_num_layers, num_edge_dim, dropout)
        self.edge_attr = Edge_Attr(num_edge_dim=num_edge_dim)
        self.MLP = MLP()
        self.embedding_1 = nn.Linear(hidden_dim+3,hidden_dim//2)
        self.embedding_2 = nn.Linear(hidden_dim//2,hidden_dim)

    def forward(self, data, edge_index):
        edge_attr_all = self.edge_attr(data["num_attr"], data["cc_attr"], data["y_init"])
        
        x = self.MLP(data["x"])
        x = torch.cat([x,edge_attr_all],dim=-1)
        b,n,d = x.shape
        x = self.embedding_1(x).reshape(b*n,-1)
        for i in range(self.trf_num_layers):
            x = self.cgnn[i](x, edge_index)
        x = self.embedding_2(x)

        return  self.predictor(x.reshape(b,n,-1))


def train(model, dataset, optimizer, batch_size, device):
    model.train()
    losses = 0
    losses1 = 0
    losses2 = 0
    losses3 = 0
    optimizer.zero_grad()
    loss_f1 = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)
    loss_f2 = torch.nn.CrossEntropyLoss(weight=city_vol_weights, ignore_index=-1)
    for data in tqdm.tqdm(
        torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=9),
        "train",
        total=len(dataset) // batch_size,
    ):
        
        data = to_var(data,device)

        pred_cc, pred_speed, pred_vol = model(data, new_edge_index)
       
        valid_count = data["edge_mask"].sum()
        loss1 = (F.mse_loss(pred_speed.squeeze(-1), data["speed_output"], reduction='none') * data["edge_mask"]).sum()/ valid_count
        loss2 = loss_f1(pred_cc.reshape(-1,3),data["y"].reshape(-1))
        loss3 = loss_f2(pred_vol.reshape(-1,3),data["volcc_output"].reshape(-1))

        loss = 0.03*loss1 + loss2 + loss3

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        losses += loss.cpu().item()
        losses1 += loss1.cpu().item()
        losses2 += loss2.cpu().item()
        losses3 += loss3.cpu().item()
    lens = len(dataset) // batch_size
    print("train_loss:{:.5f} loss_cc: {:.5f} loss_speed: {:.5f} loss_vol:{:.5f}\n".format(losses/lens,losses2/lens,losses1/lens,losses3/lens))
    return losses


@torch.no_grad()
def vaild(model, validation_dataset, batch_size, device):
    
    model.eval()

    losses = 0
    losses1 = 0
    losses2 = 0
    losses3 = 0

    loss_f1 = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)
    loss_f2 = torch.nn.CrossEntropyLoss(weight=city_vol_weights, ignore_index=-1)

    for data in tqdm.tqdm(

        torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=9),
        "vaild",
        total=len(validation_dataset) // batch_size
        ):
        data = to_var(data,device)

        pred_cc, pred_speed, pred_vol = model(data, new_edge_index)
       
        valid_count = data["edge_mask"].sum()
        loss1 = (F.mse_loss(pred_speed.squeeze(-1), data["speed_output"], reduction='none') * data["edge_mask"]).sum()/ valid_count
        loss2 = loss_f1(pred_cc.reshape(-1,3),data["y"].reshape(-1))
        loss3 = loss_f2(pred_vol.reshape(-1,3),data["volcc_output"].reshape(-1))

        loss = 0.03*loss1 + loss2 + loss3

        loss = 0.03*loss1 + loss2 + loss3
        losses += loss.cpu().item()
        losses1 += loss1.cpu().item()
        losses2 += loss2.cpu().item()
        losses3 += loss3.cpu().item()


    lens = len(validation_dataset)//batch_size
    print("valid_loss:{:.5f} loss_cc: {:.5f} loss_speed: {:.5f} loss_vol:{:.5f}\n".format(losses/lens,losses2/lens,losses1/lens,losses3/lens))


@torch.no_grad()
def test_cc(model, test_dataset, batch_size, device):
    dfs = []
    idx = 0
    model.eval()
    for data in tqdm.tqdm(torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=9),
                                "test",
                                total=len(test_dataset) // batch_size):
        # data.x = data.x.nan_to_num(-1)
        data = to_var(data,device)
        y_hat, pred_speed, pred_vol = model(data,new_edge_index)             
        df = test_dataset.torch_road_graph_mapping._torch_to_df_cc(data=y_hat.squeeze(0), day="test", t=idx)
        dfs.append(df)
        idx += 1
    df = pd.concat(dfs)
    df["test_idx"] = df["t"]
    del df["day"]
    del df["t"]
    return df

@torch.no_grad()
def test_eta(model, test_dataset, batch_size, device):
    dfs = []
    idx = 0
    for data in tqdm.tqdm(torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=9),
                                "test",
                                total=len(test_dataset) // batch_size):

        data = to_var(data,device)

        y_hat, pred_speed, pred_vol = model(data,new_edge_index) 

        df = pd.concat(
            [pd.DataFrame(data=pred_speed.squeeze(0).squeeze(-1).cpu().numpy(), columns=["pred_speed"]),
             pd.DataFrame(data=pred_vol.squeeze(0)[:, 0].cpu().numpy(), columns=["pred_vol_1"]),
             pd.DataFrame(data=pred_vol.squeeze(0)[:, 1].cpu().numpy(), columns=["pred_vol_3"]),
             pd.DataFrame(data=pred_vol.squeeze(0)[:, 2].cpu().numpy(), columns=["pred_vol_5"]),
            ],
            axis=1,
        ) 
        df["test_idx"] = [idx for i in range(y_hat.shape[1])]
        dfs.append(df)
        idx += 1
    df = pd.concat(dfs)
    return df


if __name__ == "__main__":
    t4c_apply_basic_logging_config(loglevel="DEBUG")
    # load BASEDIR from file, change to your data root
    BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)
    split = "test"
    competition = "cc"
    # competition = "eta"
    model_save_dir = Path("../checkpoints")
    model_save_dir.mkdir(exist_ok=True, parents=True)
    submission_name = "GNN_result_"+competition
    cities = ["london","melbourne","madrid"]
    city_attrs ={"london":{"nodes":59110,"edges":132414,"counters":3751,"volcc_fractions":[0.29,0.22,0.49]},
                    "madrid":{"nodes":63397,"edges":121902,"counters":3875,"volcc_fractions":[0.150,0.155,0.695]},
                    "melbourne":{"nodes":49510,"edges":94871,"counters":3982,"volcc_fractions":[0.495,0.215,0.290]},}
    if competition == "cc": 
        (BASEDIR / "submission" / submission_name ).mkdir(exist_ok=True, parents=True)
    else:
        (BASEDIR / "submissions" / submission_name ).mkdir(exist_ok=True, parents=True)



    for city  in cities:

        test_dataset = T4c22Dataset(root=BASEDIR, city=city, split=split, cachedir=Path("../data/tmp"))
        city_attr = city_attrs[city]
        
        city_class_fractions = class_fractions[city]
        city_class_weights = torch.tensor(get_weights_from_class_fractions([city_class_fractions[c] for c in ["green", "yellow", "red"]])).float()
        city_vol_weights = torch.tensor(get_weights_from_class_fractions(city_attr["volcc_fractions"])).float()

        batch_size = 1
        eval_steps = 1
        epochs = 20
        runs = 9
        dropout = 0.05
        num_edge_classes = 7
        num_node_features = 4
        num_edge_features = 32
        trf_num_layers = 6
        mlp_num_layers = 3
        trf_hidden_dim = 32
        hidden_dim = 64
        os.environ['CUDA_VISIBLE_DEVICES']= '0' 
        device = "cuda" if torch.cuda.is_available() else "cpu"

        device = torch.device(device)

        city_class_weights = city_class_weights.to(device)
        city_vol_weights = city_vol_weights.to(device)
        train_losses = defaultdict(lambda: [])
        val_losses = defaultdict(lambda: -1)

        new_edge_index  = np.load("../data/road_graph/{}/new_edge_index.npy".format(city))
        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long)
        edge_indexs = []
        for i in range(batch_size):
            x = new_edge_index+(i*city_attr["edges"])
            edge_indexs.append(x)
        new_edge_index  = torch.cat(edge_indexs,dim=-1)
        new_edge_index = new_edge_index.to(device)

        model_files = sorted(glob(f"{model_save_dir}/*{city}*"))
        
        print(city, len(model_files))

        for i, model_file in enumerate(model_files):
            j = model_file.split("_")[-1].split(".")[0]
            
            model = TrfEdgeNet(num_node_dim = num_node_features, 
                            num_edge_dim = num_edge_features, 
                            num_edge_classes = num_edge_classes,
                            trf_num_layers = trf_num_layers,
                            mlp_num_layers= mlp_num_layers,
                            trf_hidden_dim = trf_hidden_dim,
                            hidden_dim = hidden_dim)
        
            model.load_state_dict(torch.load(model_file))

            model = model.to(device)
        
            if competition == "cc":
                df_city = test_cc(model,  test_dataset = test_dataset, batch_size=batch_size, device=device)
                (BASEDIR / "submission" / submission_name / city / "labels").mkdir(exist_ok=True, parents=True)
                write_df_to_parquet(df=df_city, fn=BASEDIR / "submission" / submission_name / city / "labels" / f"cc_labels_test{j}.parquet")
                model = None
                import gc
                gc.collect()
            else:
                df_city = test_eta(model,  test_dataset = test_dataset, batch_size=batch_size, device=device)
                (BASEDIR / "submissions" / submission_name / city / "labels").mkdir(exist_ok=True, parents=True)
                write_df_to_parquet(df=df_city, fn=BASEDIR / "submissions" / submission_name / city / "labels" / f"eta_speed_test{j}.parquet")
                model = None
                import gc
                gc.collect()




