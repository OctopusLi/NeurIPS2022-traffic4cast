# -*- coding: utf-8 -*-
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
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import t4c22
from t4c22.dataloading.road_graph_mapping import TorchRoadGraphMapping
from t4c22.t4c22_config import cc_dates
from t4c22.t4c22_config import day_t_filter_to_df_filter
from t4c22.t4c22_config import day_t_filter_weekdays_daytime_only
from t4c22.t4c22_config import load_inputs
from t4c22.t4c22_config import load_basedir
import tqdm

class T4c22Competitions(Enum):
    CORE = "cc"
    EXTENDED = "eta"


class T4c22Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Path,
        city: str,
        edge_attributes=None,
        split: str = "train",
        cachedir: Optional[Path] = None,
        limit: int = None,
        day_t_filter=day_t_filter_weekdays_daytime_only,
        competition: T4c22Competitions = T4c22Competitions.CORE,
    ):
        """Dataset for t4c22 core competition (congestion classes) for one
        city.

        Get 92 items a day (last item of the day then has x loop counter
        data at 91, 92, 93, 94 and y congestion classes at 95) I.e.
        overlapping sampling, but discarding samples going over midnight.

        Missing values in input or labels are represented as nans, use `torch.nan_to_num`.
        CC labels are shift left by one in tensor as model outputs only green,yellow,red but not unclassified and allows for direct use in `torch.nn.CrossEntropy`
            # 0 = green
            # 1 = yellow
            # 2 = red


        Parameters
        ----------
        root: basedir for data
        city: "london" / "madrid" / "melbourne"
        edge_attributes: any numerical edge attribute from `road_graph_edges.parquet`
                - parsed_maxspeed
                - speed_kph
                - importance
                - oneway
                - lanes
                - tunnel
                - length_meters
        split: "train" / "test" / ...
        cachedir: location for single item .pt files (created on first access if cachedir is given)
        limit: limit the dataset to at most limit items (for debugging)
        day_t_filter: filter taking day and t as input for filtering the data. Ignored for split=="test".
        """
        super().__init__()
        self.root: Path = root
        self.cachedir = cachedir
        self.split = split
        self.city = city
        self.limit = limit
        self.day_t_filter = day_t_filter if split != "test" else None
        self.competition = competition

        self.torch_road_graph_mapping = TorchRoadGraphMapping(
            city=city,
            edge_attributes=edge_attributes,
            root=root,
            df_filter=partial(day_t_filter_to_df_filter, filter=day_t_filter) if self.day_t_filter is not None else None,
            skip_supersegments=self.competition == T4c22Competitions.CORE,
        )

        # `day_t: List[Tuple[Y-m-d-str,int_0_96]]`
        # TODO most days have even 96 (rolling window over midnight), but probably not necessary because of filtering we do.
        if split == "test":
            num_tests = load_inputs(basedir=self.root, split="test", city=city, day="test", df_filter=None)["test_idx"].max() + 1
            self.day_t = [("test", t) for t in range(num_tests)]
        else:
            self.day_t = [(day, t) for day in cc_dates(self.root, city=city, split=self.split) for t in range(4, 96) if self.day_t_filter(day, t)]

    def __len__(self) -> int:
        if self.limit is not None:
            return min(self.limit, len(self.day_t))
        return len(self.day_t)

    def __getitem__(self, idx: int) -> torch.Tensor:  # noqa:C901
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        day, t = self.day_t[idx]

        city = self.city
        basedir = self.root
        split = self.split

        # x: 4 time steps of loop counters on nodes at t=0',+15',+30',+45'
        data = None
        cache_file = self.cachedir / (f"data_{self.city}_{day}_{t}.pt" )
        if self.cachedir is not None:
            # cache_file = self.cachedir / f"inputs_{self.city}_{day}_{t}.pt"
            if cache_file.exists():
                data = torch.load(cache_file)
                return data
        x,cluster = self.torch_road_graph_mapping.load_inputs_day_t(basedir=basedir, city=city, split=self.split, day=day, t=t, idx=idx)
        speed_output,volcc_output,edge_mask = self.torch_road_graph_mapping.get_speed_volcc(basedir, city,cluster, day, t)
        y_init = self.torch_road_graph_mapping.load_y_init(basedir=basedir, city=city, cluster = cluster)

        if self.split == "test":
            # return x, None
            data = {}
            data["x"] = x
            data["num_attr"] = self.torch_road_graph_mapping.edge_attr["num_attr"].to(torch.float32)
            data["cc_attr"] = self.torch_road_graph_mapping.edge_attr["cc_attr"]
            data["y_init"] = y_init.to(torch.float32)
            return data
        
        y = self.torch_road_graph_mapping.load_labels_day_t(basedir=basedir, city=city, split=split, day=day, t=t, idx=idx)

        
        data = {}
        data["x"] = x
        data["y"] = y.nan_to_num(-1).long()
        data["num_attr"] = self.torch_road_graph_mapping.edge_attr["num_attr"].to(torch.float32)
        data["cc_attr"] = self.torch_road_graph_mapping.edge_attr["cc_attr"]
        data["speed_output"] = speed_output.to(torch.float32)
        data["volcc_output"] = volcc_output
        data["edge_mask"] = edge_mask
        data["y_init"] = y_init.to(torch.float32)
        if self.cachedir is not None:
            self.cachedir.mkdir(exist_ok=True, parents=True)
            torch.save(data, cache_file)
        
        return data


if __name__ == '__main__':
    BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)
    split = "train"
    cities = ["london"]
    batch_size = 2
    for city  in cities:
        if split == "train":
            dataset = T4c22Dataset(root=BASEDIR, city=city, split=split, cachedir=Path("data/tmp"))
        for data  in tqdm.tqdm(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8), "train", total=len(dataset) // batch_size):
            pass