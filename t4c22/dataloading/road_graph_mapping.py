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
import logging
from collections import defaultdict
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from t4c22.t4c22_config import load_cc_labels
from t4c22.t4c22_config import load_eta_labels
from t4c22.t4c22_config import load_inputs
from t4c22.t4c22_config import load_road_graph
import pyarrow.parquet as pq


class TorchRoadGraphMapping:
    def __init__(self, city: str, root: Path, df_filter, edge_attributes=None, skip_supersegments: bool = True):
        self.df_filter = df_filter

        # load road graph
        self.df_edges, df_nodes, df_supersegments = load_road_graph(root, city, skip_supersegments=skip_supersegments)
        # `edges: List[Tuple[ExternalNodeId,ExternalNodeId]]`
        self.edge_records = self.df_edges.to_dict("records")
        edges = [(r["u"], r["v"]) for r in self.edge_records]


        self.dc = df_nodes[df_nodes["counter_info"] != ""]
        # `nodes: List[ExternalNodeId]`
        nodes = [r["node_id"] for r in df_nodes.to_dict("records")]

        # enumerate nodes and edges and create mapping
        self.node_to_int_mapping = defaultdict(lambda: -1)
        for i, k in enumerate(nodes):
            self.node_to_int_mapping[k] = i

        # edge_index: Tensor of size (2,num_edges) of InternalNodeId
        self.edge_index = torch.tensor([[self.node_to_int_mapping[n] for n, _ in edges], [self.node_to_int_mapping[n] for _, n in edges]], dtype=torch.long)

        # edge_index_d: (ExternalNodeId,ExternalNodeId) -> InternalNodeId
        self.edge_index_d = defaultdict(lambda: -1)
        self.edges = edges
        self.nodes = nodes
        for i, (u, v) in enumerate(edges):
            self.edge_index_d[(u, v)] = i

        # sanity checking edges and nodes are unique
        assert len(edges) == len(set(edges)), (len(edges), len(set(edges)))
        assert len(nodes) == len(set(nodes)), (len(nodes), len(set(nodes)))

        # sanity checking edge_index and edge_index_d size coincide with number of edges
        # beware, after accessing
        assert len(self.edge_index_d) == len(edges), (len(self.edge_index_d), len(edges))
        assert self.edge_index.size()[1] == len(edges), (self.edge_index.size()[1], len(edges))
        assert self.edge_index.size()[1] == len(self.edge_index_d), (self.edge_index.size()[1], len(self.edge_index_d))

        # sanity checking node_to_int_mapping has size number of nodes
        assert len(self.node_to_int_mapping) == len(nodes), (len(self.node_to_int_mapping), len(nodes))
        
        # edge_attr
        self.edge_attributes = edge_attributes
        self.edge_attr = self.get_edge_attr(self.df_edges)


        self.y_flow = torch.from_numpy(np.array(self.df_edges["flow"].values,dtype=float))

        snapshot_file = root / 'snapshots_cc' / f'cc_volume_cluster_baseline_exp_10_clusters_{city}.parquet'
        self.counts_cc_df = pd.read_parquet(snapshot_file)



    def get_edge_attr(self,df_edges):
        df = pd.DataFrame(df_edges,columns = ["parsed_maxspeed", "importance", "length_meters", "counter_distance","oneway",'tunnel','counter_distance','lanes','flow','limit_speed'])
        edge_attr = {}
        for key in ["parsed_maxspeed",'flow' , "length_meters"]:
            mean = df[key].mean()
            std = df[key].std()
            df[key] = df[key].apply(lambda x: (x - mean) / std)
        num_attr = torch.from_numpy(np.array(df[["parsed_maxspeed",'flow' , "length_meters","counter_distance","limit_speed"]].values,dtype=float))
        edge_attr["num_attr"] = num_attr
        cc_attr = torch.from_numpy(np.array(df[["importance","oneway",'tunnel','lanes']].values,dtype=float)).long()
        edge_attr["cc_attr"] = cc_attr
        return edge_attr

    def load_y_init(self,basedir: Path, city: str, cluster:int)-> torch.Tensor:
        df_e = self.df_edges.copy()
        df_e['cluster'] = np.array([cluster]*len(df_e))
        df_e = df_e.merge(self.counts_cc_df, on=["u", "v",'cluster'], how="left")
        df_e = df_e.fillna(0.333)
        y_init = df_e[['logit_green','logit_yellow','logit_red']].values      
        y_init = torch.from_numpy(np.array(y_init,dtype=float))
        return y_init

    def get_speed_volcc(self,basedir: Path, city: str,cluster: int, day:str,t:int):
        
        if day == "test":
            speed_output = None
            volcc_output = None
            edge_mask = None
        else:
            infix = "" if day is None else f"_{day}"
            fn = basedir /'speed_classes'/ city /f"speed_classes{infix}.parquet"
            df_edge = self.df_edges[['u','v']].copy()
            sp_output = pd.read_parquet(fn)
            sp_output = sp_output[sp_output['t']==t]
            sp_output = df_edge.merge(sp_output,on=['u','v'],how='left')
            sp_output['mask'] = sp_output['u'].apply(lambda x: 1)
            sp_output.iloc[pd.isna(sp_output["volume_class"]),-1] = 0
            sp_output["volume_class"].fillna(-1,inplace=True)
            sp_output["median_speed_kph"].fillna(0,inplace=True)
            sp_output["volume_class"] = sp_output["volume_class"].apply(lambda x :int((x+1)/2-1))
            speed_output = torch.from_numpy(np.array(sp_output["median_speed_kph"].values,dtype=float))
            volcc_output = torch.from_numpy(np.array(sp_output["volume_class"].values,dtype=int)).long()
            edge_mask = torch.from_numpy(np.array(sp_output["mask"].values,dtype=int)).long()

        return speed_output,volcc_output,edge_mask

    def load_inputs_day_t(self, basedir: Path, city: str, split: str, day: str, t: int, idx: int) -> torch.Tensor:
        """Used by dataset getter to load input data (sparse loop counter data
        on nodes) from parquet into tensor.

        Parameters
        ----------
        basedir: data basedir see `README`
        city: "london"/"madrid"/"melbourne"
        split: "train"/"test"/...
        day: date
        t: time of day in 15-minutes in range [0,....96)
        idx: dataset index

        Returns
        -------
        Tensor of size (number-of-nodes,4).
        """
        input_attrs = {'london': {'mean': 387.6885766542457, 'std': 304.3742438740916}, 
                    'melbourne': {'mean': 236.04950022933497, 'std': 234.39031407078073}, 
                    'madrid': {'mean': 535.3109241594892, 'std': 865.938607714607}}
        input_attr = input_attrs[city]
        df_x = load_inputs(basedir, city=city, split=split, day=day, df_filter=self.df_filter)

        if day == "test":
            data = df_x[(df_x["test_idx"] == idx)].copy()
        else:
            data = df_x[(df_x["day"] == day) & (df_x["t"] == t)].copy()
            
        cluster = data['cluster'].values[0]
        data['vol'] = np.array(data['volumes_1h'].to_numpy().tolist())[:,3]    
        data = self.dc.merge(data,on=["node_id"],how = "left")
        x = torch.tensor(data['vol'].values).float()
        x = x.nan_to_num(0)
        x = (x-input_attr['mean'])/input_attr['std']

        return x,cluster


    def load_labels_day_t(self, basedir: Path, city: str, split: str, day: str, t: int, idx: int) -> torch.Tensor:
        """Used by dataset getter to load congestion class labels (sparse
        congestion classes) from parquet into tensor.

        Parameters
        ----------
        basedir: data basedir see `README`
        city: "london"/"madrid"/"melbourne"
        split: "train"/"test"/...
        day: date
        t: time of day in 15-minutes in range [0,....96)
        idx: dataset index


        Returns
        -------
        Float tensor of size (number-of-edges,), with edge congestion class and nan if unclassified.
        """
        df_y = load_cc_labels(basedir, city=city, split=split, day=day, with_edge_attributes=True, df_filter=self.df_filter)
        if day == "test":
            data = df_y[(df_y["test_idx"] == idx)]
        else:
            data = df_y[(df_y["day"] == day) & (df_y["t"] == t)]

        y = self._df_cc_to_torch(data)

        if len(data) == 0:
            logging.warning(f"{split} {city} {(idx, day, t)} no classified")
        return y

    def load_eta_labels_day_t(self, basedir: Path, city: str, split: str, day: str, t: int, idx: int) -> torch.Tensor:
        """Used by dataset getter to load eta (sparse) on supersegments from
        parquet into tensor.

        Parameters
        ----------
        basedir: data basedir see `README`
        city: "london"/"madrid"/"melbourne"
        split: "train"/"test"/...
        day: date
        t: time of day in 15-minutes in range [0,....96)
        idx: dataset index


        Returns
        -------
        Float tensor of size (number-of-supersegments,), with supersegment eta and nan if unavailable.
        """
        df_y = load_eta_labels(basedir, city=city, split=split, day=day, df_filter=self.df_filter)
        if day == "test":
            data = df_y[(df_y["test_idx"] == idx)]
        else:
            data = df_y[(df_y["day"] == day) & (df_y["t"] == t)]

        y = self._df_eta_to_torch(data)

        if len(data) == 0:
            logging.warning(f"{split} {city} {(idx, day, t)} no classified")
        return y

    def _df_cc_to_torch(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Parameters
        ----------
        data: data frame for (day,t) with columns "u", "v", "cc".

        Returns
        -------
        Float tensor of size (number-of-edges,), containing edge congestion class and nan if unclassified.
        """
        y = torch.full(size=(len(self.edges),), fill_value=float("nan"))
        if len(data[data["cc"] > 0]) > 0:
            data = data[data["cc"] > 0].copy()
            assert len(data) <= len(self.edges)
            data["edge_index"] = [self.edge_index_d[u, v] for u, v in zip(data["u"], data["v"])]

            # sanity check as defaultdict returns -1 for non-existing edges
            assert len(data[data["edge_index"] < 0]) == 0
            assert data["cc"].min() >= 1, (data["cc"].min(), data)
            assert data["cc"].max() <= 3, (data["cc"].max(), data)

            # shift left by one in tensor as model outputs only green,yellow,red but not unclassified!
            # 0 = green
            # 1 = yellow
            # 2 = red
            data["cc"] = data["cc"] - 1
            y[data["edge_index"].values] = torch.tensor(data["cc"].values).float()
        return y

    def _torch_to_df_cc(self, data: torch.Tensor, day: str, t: int) -> pd.DataFrame:
        """
        Parameters
        ----------
        Float tensor of size (number-of-edges,3) with logits for green, yellow and red.

        Returns
        -------
        Data frame for (day,t) with columns "u", "v", "day", "t", "logit_green", "logit_yellow", "logit_red".
        """

        froms = [t[0] for t in self.edges]
        tos = [t[1] for t in self.edges]
        df = pd.concat(
            [
                pd.DataFrame(data=data[:, 0].cpu().numpy(), columns=["logit_green"]),
                pd.DataFrame(data=data[:, 1].cpu().numpy(), columns=["logit_yellow"]),
                pd.DataFrame(data=data[:, 2].cpu().numpy(), columns=["logit_red"]),
            ],
            axis=1,
        )
        df["u"] = froms
        df["v"] = tos
        df["day"] = day
        df["t"] = t
        return df

    def _df_eta_to_torch(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Parameters
        ----------
        data: data frame for (day,t) with columns "identifier", "eta".

        Returns
        -------
        Float tensor of size (number-of-supersegments,), containing etas and nan if undefined
        """
        y = torch.full(size=(len(self.supersegments),), fill_value=float("nan"))
        if len(data) > 0:
            assert len(data) <= len(self.supersegments)
            data["supersegment_index"] = [self.supersegments_d[identifier] for identifier in data["identifier"]]
            y[data["supersegment_index"].values] = torch.tensor(data["eta"].values).float()
        return y

    def _torch_to_df_eta(self, data: torch.Tensor, day: str, t: int) -> pd.DataFrame:
        """
        Parameters
        ----------
        Float tensor of size (number-of-supersegments,) with etas.

        Returns
        -------
        Data frame for (day,t) with columns "identifier", "day", "t", "eta".
        """

        df = pd.DataFrame(data=data.cpu().numpy(), columns=["eta"])
        df["identifier"] = self.supersegments
        df["day"] = day
        df["t"] = t
        return df
