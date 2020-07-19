
# reid project 

## dataset=market1
kaggle name = reid-custom
|                                        method                                         | rank@1 |  Map   |  kaggle version  |
| :-----------------------------------------------------------------------------------: | :----: | :----: | :--------------: |
|                            [pcb](project_result/pcb.ipynb)                            | 93.23% | 79.69% | reid-custom-v60  |
|                      [pcb_lr](project_result/pcb_same_lr.ipynb)                       | 89.58% | 72.16% | reid-custom-v76  |
|              [pcb_sga_s_img(256x128)](project_result/pcb_same_lr.ipynb)               | 84.74% | 66.00% | reid-custom-v82  |
|               [pcb_img(256x128)](project_result/pcb_img(256x128).ipynb)               | 87.89% | 72.34% | reid-custom-v83  |
|              [pcb_sga_s_lr(layers)](project_result/pcb_lr(layers).ipynb)              | 50.36% | 29.62% | reid-custom-v87  |
| [pcb_img(256x128)_label_smooths](project_result/pcb_img(256x128)_label_smooths.ipynb) | 85.78% | 68.20% | reid-custom-v94  |
|        [pcb_img(256x128)_sga_v2](project_result/pcb_img(256x128)_sga_v2.ipynb)        | 88.09% | 72.52% | reid-custom-v97  |
|        [pcb_img(384x128)_sga_v2](project_result/pcb_img(384x128)_sga_v2.ipynb)        | 89.70% | 74.61% | reid-custom-v98  |
|             [pcb(384x128)_sga_s](project_result/pcb(384x128)_sga_s.ipynb)             | 93.02% | 79.58% | reid-custom-v125 |
|                                     [pcb(384x128]                                     | 92.96% | 79.92% | reid-custom-v150 |
|                              [pcb(384x128_global_parts]                               | 92.73% | 79.94% | reid-custom-v152 |
|                         [pcb(384x128_global(att_c)_parts{v5}]                         | 92.10% | 78.15% | reid-custom-v153 |
|                         [pcb(384x128_global(att_s)_parts{v5}]                         | 91.69% | 77.73% | reid-custom-v154 |
|                         [pcb(256x128_global(att_c)_parts{v5}]                         | 91.63% | 78.32% | reid-custom-v156 |
|                         [pcb(256x128_global(att_s)_parts{v5}]                         | 91.51% | 78.06% | reid-custom-v155 |





##  dataset=occluded_reid (train in market1,and test in occluded_reid)
kaggle name = reid-custom
|                                        method                                         | rank@1 |  Map   |  kaggle version  |
| :-----------------------------------------------------------------------------------: | :----: | :----: | :--------------: |
|                            [pcb](project_result/pcb.ipynb)                            | 57.40% | 52.29% | reid-custom-v60  |
|                      [pcb_lr](project_result/pcb_same_lr.ipynb)                       | 45.20% | 40.13% | reid-custom-v76  |
|              [pcb_sga_s_img(256x128)](project_result/pcb_same_lr.ipynb)               | 48.90% | 45.72% | reid-custom-v82  |
|               [pcb_img(256x128)](project_result/pcb_img(256x128).ipynb)               | 59.00% | 54.28% | reid-custom-v83  |
|              [pcb_sga_s_lr(layers)](project_result/pcb_lr(layers).ipynb)              | 33.90% | 28.66% | reid-custom-v87  |
| [pcb_img(256x128)_label_smooths](project_result/pcb_img(256x128)_label_smooths.ipynb) | 51.90% | 48.22% | reid-custom-v94  |
|        [pcb_img(256x128)_sga_v2](project_result/pcb_img(256x128)_sga_v2.ipynb)        | 54.90% | 50.28% | reid-custom-v97  |
|        [pcb_img(384x128)_sga_v2](project_result/pcb_img(384x128)_sga_v2.ipynb)        | 59.50% | 53.03% | reid-custom-v98  |
|             [pcb(384x128)_sga_s](project_result/pcb(384x128)_sga_s.ipynb)             | 60.20% | 53.88% | reid-custom-v125 |
|                                     [pcb(384x128]                                     | 60.50% | 54.14% | reid-custom-v150 |
|                              [pcb(384x128_global_parts]                               | 58.20% | 52.85% | reid-custom-v152 |
|                         [pcb(384x128_global(att_c)_parts{v5}]                         | 53.40% | 47.54% | reid-custom-v153 |
|                         [pcb(384x128_global(att_s)_parts{v5}]                         | 56.00% | 50.14% | reid-custom-v154 |
|                         [pcb(256x128_global(att_c)_parts{v5}]                         | 54.80% | 49.23% | reid-custom-v156 |
|                         [pcb(256x128_global(att_s)_parts{v5}]                         | 49.70% | 46.46% | reid-custom-v155 |



