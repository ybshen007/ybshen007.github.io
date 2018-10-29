---
layout:     post
title:      "Ubuntu装机记录"
date:       2018-10-28
author:     "Shi Lou"
catalog: true
categories: linux
tags:
    - Linux
    - 装机
    - 配置
---

> 前两天不慎摔坏了笔记本硬盘，换了一块新的ssd，需要重新装系统软件及进行相关配置，在此做一个记录。
<!--more-->

*环境：Ubuntu16.04*

装系统就不具体展开了，注意的是我装了英文版，因为如果装中文版的话，用户目录下的文件夹会初始化成中文名，终端中敲中文有点麻烦，需要后期更改。因此就直接装英文版了～

## 更换源
由于墙的原因，Ubuntu官方源不怎么好使，先换成阿里云的源。
```sh
sudo vi /etc/apt/sources.list
```
```sh
deb http://mirrors.aliyun.com/ubuntu/ xenial main restricted universe multiverse 
deb http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted universe multiverse 
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted universe multiverse 
deb http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse 
deb http://mirrors.aliyun.com/ubuntu/ xenial-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial main restricted universe multiverse 
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted universe multiverse 
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted universe multiverse 
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse 
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-proposed main restricted universe multiverse 
deb http://archive.canonical.com/ubuntu/ xenial partner 
```
```sh
sudo apt-get update
```

## 搜狗输入法
第一件事肯定是先装个中文输入法，方便后续工作。Ubuntu下搜狗输入法是一个比较好的选择（可能也是唯一？）。
先在官网下deb包，[链接](https://pinyin.sogou.com/linux/?r=pinyin)。
cd到deb包所在的目录，安装：
```sh
sudo apt-get install -f
sudo dpkg -i sogou*.deb
```
设置系统默认输入法由iBus改为fcitx：
system setting-->language support-->下拉iBus改为fcitx
注销，重新登录。
右键右上角键盘图标，进入设置，点左下角加号，添加sogou输入法。

ps: 如果出现两个搜狗拼音的图标，解决方法
```sh
sudo apt-get remove fcitx-ui-qimpanel
```

## ssh
这个比较简单
```sh
sudo apt-get install openssh-server
```
装好默认会启动，如果没有启动可以按以下命令启动
```sh
sudo /etc/init.d/ssh start
```
查看是否启动
```sh
ps -e | grep ssh
```
如果能看到sshd 就说明已经启动。

## Chrome
Ubuntu自带的是Firefox，而我习惯了Chrome，因此Chrome是必备的。
首先添加google软件的源跟验证公钥，并更新源。
```sh
sudo wget http://www.linuxidc.com/files/repo/google-chrome.list -P /etc/apt/sources.list.d/
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub  | sudo apt-key add -
sudo apt-get update
```
然后安装：
```sh
sudo apt-get install google-chrome-stable
```
安装成功后可以通过以下命令启动：
```sh
/usr/bin/google-chrome-stable
```
将其锁定到launcher就可以单击启动了，可以跟FireFox说byebye了～

## Shadowsocks
由于众所周知的原因，ss也是必须的。
### 安装ss
```sh
sudo apt-get install python-pip
sudo apt-get install python-setuptools m2crypto
sudo pip install shadowsocks
```
如果ss加密方式使用的是chacha-20-ietf-poly1305，由于以上安装方式并不支持该加密方式，需要安装libsodium-dev：
```sh
sudo apt install libsodium-dev
sudo pip install https://github.com/shadowsocks/shadowsocks/archive/master.zip -U
```
可以参考[这里](https://github.com/shadowsocks/shadowsocks/issues/1046)。

**ps: 以上安装shadowsocks的两条命令都用了sudo，是为了方便后续配置自动启动，如果不安装在root下，后续配置会比较麻烦。**

启动：
推荐使用配置文件的方式启动，一劳永逸，配置文件shadowsocks.json如下：
```json
{
    "server": "",       //服务器ip
    "server_port": ,    //端口
    "local_port": 1080, //本地端口
    "password": "",     //密码
    "timeout": 600,
    "method": "chacha-20-ietf-poly1305" //加密方式
}
```
```sh
sudo sslocal -c ~/Documents/shadowsocks.json
```

### 设置开机自动启动
```sh
sudo vim /etc/rc.local
```
在exit 0前添加启动命令：
```sh
sudo sslocal -c /home/**/Documents/shadowsocks.json &
```

**ps: 这里最后的&不要省略，去掉&的话启动时会等这条命令执行完再去执行后面的命令，如果这里卡住了就开不了机了。一个坑是装了Gnome后开机会一直转圈**

### 配置浏览器
以上只是安装了ss，还需要在浏览器中进行配置，才可以科学上网。
首先要安装SwitchOmega插件，[下载地址](https://github.com/FelisCatus/SwitchyOmega/releases/)。
下载完后拖到Chrome插件页面chrome://extensions/就行了（如果拖曳安装失败，打开插件页面的开发者模式，重启Chrome即可）。

进入SwitchOmega配置页面，新建情景模式，名字随便取，选择代理服务器（Proxy Profile）。
![new_profile](/images/new_profile.png)

代理协议选择SOCKS5，代理服务器127.0.0.1，端口1080，保存。
![profile](/images/profile.png)

设置自动切换，让不需要翻墙的流量走直连，选左边的自动切换，规则列表网址中填https://raw.githubusercontent.com/gfwlist/gfwlist/master/gfwlist.txt， 立即更新情景模式，可以看到一大批无法访问的网站会被添加到规则列表里。在切换规则中选择刚才的情景，保存即可。
![auto_switch.png](/images/auto_switch.png)

### 配置全局ss
通过以上配置，可以在浏览器中科学上网，但有时候要在终端中科学上网，比如git如果不用科学上网的话，push和pull会很慢。

安装privoxy：
```sh
sudo apt install privoxy
```

配置privoxy：
```sh
sudo vim /etc/privoxy/config
```
去掉1336行注释，监听端口1080，注意最后有个点。
```sh
forward-socks5t / 127.0.0.1:1080 .
```
监听默认8118端口
```sh
listen-address localhost:8118
```
设置http和https的全局代理，在环境变量中加入下面两行
```sh
export http_proxy=http://localhost:8118
export https_proxy=https://localhost:8118
```
应用生效
```sh
source /etc/profile
```
启动服务
```sh
sudo service privoxy start
```
可以通过curl来测试是否成功开启全局ss：
```sh
curl www.google.com
```
如果开启成功，会返回google页面，失败的话没反应。

**ps: 之后终端中默认是科学上网的，如果要下载大容量的文件，例如百度云上的，为了节约流量，同时国内网站科学上网反而为降低速度，注释掉配置文件/etc/privoxy/config中的1336行即可**

## Gnome
Gnome安装配置比较简单，但是不推荐使用，存在一些问题，theme也没有unity桌面的好看。
```sh
sudo apt-get install ubuntu-gnome-desktop
```
会出现一个图形界面，两个选项选任意一个都行，装完后重启，登录时选择Gnome桌面即可。原来的桌面可以删除：
```sh
sudo apt-get remove ubuntu-desktop
```

**ps: 这里的一个坑是，我装了Gnome后，之前开机自启的sslocal进程变得无效，即进程正常启动了，但是不能科学上网。**
**解决办法：**
试了很多种都没用，最后只能创建一个sslocal的快捷方式，然后把它放到Gnome的开机自启中去。

创建启动脚本
```sh
sudo vim /usr/local/bin/autostart_sslocal
```
```sh
#! /bin/bash
/usr/local/bin/sslocal -c /home/**/Documents/shadowsocks.json
```

创建快捷方式
```sh
sudo vim /usr/share/applications/sslocal.desktop
```
```sh
[Desktop Entry]
Encoding=UTF-8
Version=1.0
Name=sslocal
GenericName=sslocal
Comment=sslocal
Exec=/usr/local/bin/autostart_sslocal
Terminal=false
Type=Application
X-Desktop-File-Install-Version=0.22
```

加入启动项
```sh
ln -s /usr/share/applications/sslocal.desktop ~/.config/autostart/sslocal.desktop
```

## Gnome-theme
启动tweak-tool，在扩展中打开user-theme选项即可。

我使用的配置：
主题： [arc](https://github.com/horst3180/arc-theme)
扩展：
Dash to Dock
Hide Top Bar
User Themes

以上是Gnome主题的配置，Unity的配置为：
主题
```sh
sudo add-apt-repository ppa:noobslab/themes 
sudo apt-get update 
sudo apt-get install flatabulous-theme 
```
图标
```sh
sudo add-apt-repository ppa:noobslab/icons 
sudo apt-get update 
sudo apt-get install ultra-flat-icons
```

## 字体
[maonaco-font-master.zip](https://pan.baidu.com/s/1fgykIDfL8dwFYUup3r7V6w)
解压后cd到目录中安装：
```sh
sudo ./install-font-ubuntu.sh https://github.com/todylu/monaco.ttf/blob/master/monaco.ttf?raw=true

```
安装完后用unity-tweak-tool工具替换。

## zsh
### 安装
安装zsh
```sh
sudo apt-get install zsh
```

安装oh-my-zsh
```sh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```

安装完重启后打开终端即自动进入zsh。

### 配置
主题：ys
```sh
vim ~/.zshrc
set ZSH_THEME="ys"
source ~/.zshrc
```

扩展:
终端高亮
```sh
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
vim ~/.zshrc
set plugins=([plugins] zsh-syntax-highlighting)
source ~/.zshrc
```

自动补全可能路径

```sh
git clone git://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
vim ~/.zshrc
set plugins=([plugins] zsh-autosuggestions)
source ~/.zshrc
```

## vim

[配置文件](https://github.com/ybshen007/resources/blob/master/.vimrc)


## Tensorflow

```sh
pip install --ignore-installed --upgrade url
```

Python 2.7

CPU only:
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp27-none-linux_x86_64.whl

GPU support:
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp27-none-linux_x86_64.whl
 

Python 3.4

CPU only:
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp34-cp34m-linux_x86_64.whl　

GPU support:
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp34-cp34m-linux_x86_64.whl
 

Python 3.5

CPU only:
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp35-cp35m-linux_x86_64.whl

GPU support:
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp35-cp35m-linux_x86_64.whl
 

Python 3.6

CPU only:
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl　

GPU support:
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp36-cp36m-linux_x86_64.whl