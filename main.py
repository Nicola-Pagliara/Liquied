import json

from EDA import EDA as eda
from Train import Train as trn
from Evaluation import Eval as eval
from Models import Clustering as clus


def main():
    eda.overview_analysis()
    trn.train()
    eval.test_phase()
    clus.dbscan_encoded()


if __name__ == '__main__':
    main()
