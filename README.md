## TORS_PING

Experiments codes for the paper:

Dugang Liu, Shenxian Xian, Yuhao Wu, Xiaolian Zhang, Zhong Ming. Pairwise Intent Graph Embedding Learning for Context-aware Recommendation with Knowledge Graph. Accepted by ACM TORS.

**Please cite our ACM TORS paper if you use our codes. Thanks!**

## Requirement

- python==3.6.9
- tensorflow==1.15.3+nv

## Usage

* Our implementation references UEG ([Link](https://github.com/dgliu/KDD22_UEG)) and KGIN ([Link](https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network)).
* We use v1 and v2 to identify the variants in the fusion strategy and the integration operation, respectively. We also use different suffixes to distinguish different improved strategies. For more information, please refer to Sections 5.4 and 5.5 of our paper.
* For different data sets, some command line examples are as follows:

**For Amazon-Book:**

```bash
python PING_v2_WF.py --dataset Amazon-Book --num_gcn_layers 2 --reg 1e-3 --decoder_type FM --adj_norm_type ls --num_negatives 4 --intent_weight 0.7 --test_interval 5 --stop_cnt 10 --intent_weight=0.1 --gpu_id=2
```

**For LFM:**

```bash
python PING_v1_SA.py --dataset LFM --num_gcn_layers 2 --reg 1e-3 --decoder_type FM --adj_norm_type ls --num_negatives 4 --intent_weight 0.1 --test_interval 5 --stop_cnt 10 --gpu_id=5
```

**For Yelp:**

```bash
python PING_v1_WF.py --dataset Yelp --num_gcn_layers 2 --reg 1e-3 --decoder_type FM --adj_norm_type ls --num_negatives 4 --intent_weight 0.9 --test_interval 5 --stop_cnt 10 --gpu_id=4 --weight_11=0.1
```


## 

If you have any issues or ideas, feel free to contact us ([dugang.ldg@gmail.com](mailto:dugang.ldg@gmail.com)).
