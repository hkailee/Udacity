#!/bin/bash
#git rm -r --cached .
#git filter-branch --tree-filter 'rm -rf deep-learning/projects/other_mini_projects/cycleGAN/summer2winter_yosemite.zip' HEAD
#git filter-branch --tree-filter 'rm -rf deep-learning/projects/other_mini_projects/DCGAN_SVHN/data/train_32x32.mat' HEAD
git add .
git commit -m "update mini-project"
git push
