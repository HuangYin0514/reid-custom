
# reid project 

## dataset=market1
kaggle name = reid-custom
|                           method                           | rank@1 |  Map   | kaggle version  |
| :--------------------------------------------------------: | :----: | :----: | :-------------: |
|              [pcb](project_result/pcb.ipynb)               | 93.23% | 79.69% | reid-custom-v60 |
|         [pcb_lr](project_result/pcb_same_lr.ipynb)         | 89.58% | 72.16% | reid-custom-v76 |
| [pcb_sga_s_img(256x128)](project_result/pcb_same_lr.ipynb) | 84.74% | 66.00% | reid-custom-v82 |



##  dataset=occluded_reid (train in market1,and test in occluded_reid)
kaggle name = reid-custom
|                           method                           | rank@1 |  Map   | kaggle version  |
| :--------------------------------------------------------: | :----: | :----: | :-------------: |
|              [pcb](project_result/pcb.ipynb)               | 57.40% | 52.29% | reid-custom-v60 |
|          [pcb](project_result/pcb_same_lr.ipynb)           | 45.20% | 40.13% | reid-custom-v76 |
| [pcb_sga_s_img(256x128)](project_result/pcb_same_lr.ipynb) | 48.90% | 45.72% | reid-custom-v82 |


