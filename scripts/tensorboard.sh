#!/bin/bash

rm -rf /home/hy/vscode/reid-custom/log/tensorboard/log*

mv /mnt/c/Users/10713/Downloads/logs.tar /home/hy/vscode/reid-custom/log/tensorboard/logs.tar

tar zxf /home/hy/vscode/reid-custom/log/tensorboard/logs.tar

tensorboard --logdir=/home/hy/vscode/reid-custom/log/


