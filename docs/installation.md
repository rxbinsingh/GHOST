# Installation

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, CPU also works)

## Install from source

```bash
git clone https://github.com/rxbinsingh/GHOST.git
cd GHOST
pip install -r requirements.txt
```

## Google Colab (quickest)

```python
!pip install torch networkx scikit-learn pandas numpy matplotlib scipy -q
```

Then paste `src/ghost_complete.py` into a cell and run.

## Real Dataset (optional)

Download the DataCo Smart Supply Chain dataset from Kaggle:
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

File: `DataCoSupplyChainDataset.csv`

Then run:
```python
report = main('/content/DataCoSupplyChainDataset.csv')
```
