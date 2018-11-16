---
title: Ubuntu环境下pyrouge+ROUGE安装
date: 2018-11-16 14:08:12
autohr: "Shi Lou"
categories: linux
tags: 
	- Linux
	- 配置
	- NLP
---

> 记录Ubuntu环境下ROUGE+pyrouge的配置过程。
<!-- more -->

# 1. ROUGE
## 1.1 下载安装
### 1.1.1 下载依赖
```bash
sudo cpan install DB_file
sudo cpan install XML::DOM
```
### 1.1.2 下载ROUGE
下载ROUGE-1.5.5: 链接: https://pan.baidu.com/s/1OhC1NmQcLEMoNGTnksVaXQ 提取码: uyk6 
解压到~/ROUGE，添加环境变量
```bash
vim ~/.profile
export ROUGE_EVAL_HOME="~/ROUGE/data"
source ~/.profile
```
### 1.1.3 测试
```bash
cd ~/ROUGE
./runROUGE-test.pl
```
出现以下信息表示安装成功：
```bash
../ROUGE-1.5.5.pl -e ../data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a ROUGE-test.xml > ../sample-output/ROUGE-test-c95-2-1-U-r1000-n4-w1.2-a.out
../ROUGE-1.5.5.pl -e ../data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m ROUGE-test.xml > ../sample-output/ROUGE-test-c95-2-1-U-r1000-n4-w1.2-a-m.out
../ROUGE-1.5.5.pl -e ../data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m -s ROUGE-test.xml > ../sample-output/ROUGE-test-c95-2-1-U-r1000-n4-w1.2-a-m-s.out
../ROUGE-1.5.5.pl -e ../data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -l 10 -a ROUGE-test.xml > ../sample-output/ROUGE-test-c95-2-1-U-r1000-n4-w1.2-l10-a.out
../ROUGE-1.5.5.pl -e ../data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -l 10 -a -m ROUGE-test.xml > ../sample-output/ROUGE-test-c95-2-1-U-r1000-n4-w1.2-l10-a-m.out
../ROUGE-1.5.5.pl -e ../data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -l 10 -a -m -s ROUGE-test.xml > ../sample-output/ROUGE-test-c95-2-1-U-r1000-n4-w1.2-l10-a-m-s.out
../ROUGE-1.5.5.pl -e ../data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -b 75 -a ROUGE-test.xml > ../sample-output/ROUGE-test-c95-2-1-U-r1000-n4-w1.2-b75-a.out
../ROUGE-1.5.5.pl -e ../data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -b 75 -a -m ROUGE-test.xml > ../sample-output/ROUGE-test-c95-2-1-U-r1000-n4-w1.2-b75-a-m.out
../ROUGE-1.5.5.pl -e ../data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -b 75 -a -m -s ROUGE-test.xml > ../sample-output/ROUGE-test-c95-2-1-U-r1000-n4-w1.2-b75-a-m-s.out
../ROUGE-1.5.5.pl -e ../data -3 HM -z SIMPLE DUC2002-BE-F.in.26.lst 26 > ../sample-output/DUC2002-BE-F.in.26.lst.out
../ROUGE-1.5.5.pl -e ../data -3 HM DUC2002-BE-F.in.26.simple.xml 26 > ../sample-output/DUC2002-BE-F.in.26.simple.out
../ROUGE-1.5.5.pl -e ../data -3 HM -z SIMPLE DUC2002-BE-L.in.26.lst 26 > ../sample-output/DUC2002-BE-L.in.26.lst.out
../ROUGE-1.5.5.pl -e ../data -3 HM DUC2002-BE-L.in.26.simple.xml 26 > ../sample-output/DUC2002-BE-L.in.26.simple.out
../ROUGE-1.5.5.pl -e ../data -n 4 -z SPL DUC2002-ROUGE.in.26.spl.lst 26 > ../sample-output/DUC2002-ROUGE.in.26.spl.lst.out
../ROUGE-1.5.5.pl -e ../data -n 4 DUC2002-ROUGE.in.26.spl.xml 26 > ../sample-output/DUC2002-ROUGE.in.26.spl.out
```

# 2. pyrouge
pyrouge要安装最新版，否则测试时会出现error和failure。
```bash
git clone https://github.com/bheinzerling/pyrouge
cd pyrouge
python setup.py install
```
指定之前ROUGE位置：
```bash
pyrouge_set_rouge_path ~/ROUGE
```
测试：
```bash
python -m pyrough.test
```
出现ok即安装成功。

# 参考资料
1. [WORKING WITH ROUGE 1.5.5 EVALUATION METRIC IN PYTHON](https://ireneli.eu/2018/01/11/working-with-rouge-1-5-5-evaluation-metric-in-python/)
2. [Unittests fail](https://github.com/bheinzerling/pyrouge/issues/7)
