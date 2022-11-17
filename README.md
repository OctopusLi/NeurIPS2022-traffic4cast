# Traffic4cast2022
## Data Preparation
After downloading the [data] (https://github.com/iarai/NeurIPS2022-traffic4cast)) and [checkpoints](https://drive.google.com/drive/folders/1-iqirakUhSjWoiKy-MRY5eRGPswE6F9-), run
```bash
prepare_training_data_cc.py -d <data folder with unzipped downloads>
prepare_training_data_eta.py -d <data folder with unzipped downloads>
prepare_training_check_labels.py -d <data folder with unzipped downloads>
cd data
run data_preprocess.ipynb
```

### Usage

```bash
cd model
run cluster.ipynb
python GNN_model_train.py
python GNN_model_test.py
python submission_cc.py
python submission_eta.py
```

### Acknowledgements
This repository is based on [NeurIPS2022-Traffic4cast](https://github.com/iarai/NeurIPS2022-traffic4cast) from the Institute of Advanced Research in Artificial Intelligence (IARAI).