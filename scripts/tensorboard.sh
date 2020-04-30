#!/bin/bash

rm -rf ./log*

mv /mnt/c/Users/10713/Downloads/logs.tar ./logs.tar

tar zxf logs.tar

tensorboard --logdir=./
