#!/bin/bash

# train 
python /kaggle/working/reid-custom/projects/PCB/main.py \
--config-file /kaggle/working/reid-custom/projects/PCB/pcb_p6.yaml \
--root /kaggle/input/


# test
python projects/PCB/main.py \
--config-file  projects/PCB/pcb_p6_test.yaml \
--model-resume /home/hy/vscode/reid-custom/log/pcb_p6_save/model/model.pth.tar-60

# test kaggle
!python projects/PCB/main.py \
--config-file  projects/PCB/pcb_p6_test.yaml \
--model-resume /kaggle/working/log/pcb_p6_save/model/model.pth.tar-60 \
--root /kaggle/input/