# article_image
首先完成mcnn的结果复现

安装pycocotools
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python3 setup.py build
python3 setup.py install

安装必要的包
pip3 install -r requirements.txt

下载数据文件
chmod +x download.sh
./download.sh

预处理数据, 需指定数据文件路径
python3 build_vocab.py
python3 resize.py

linux需安装tk
sudo apt-get install python3-tk

nltk需要下载数据
import nltk
nltk.download("punkt")


比如图片为
100
500
15 0000

18s 10000
40 * 18s = 720s 
10000/18s = 555 pair/s

100张图
500个caption

提取出100个imgid，
根据imgid提取出500个captionid
（如果imgid对应的不是5个caption，就下一个）

取出imgid对应的img（是dict），放到list里，
对每个img，根据地址读取img数据，并且transform，然后存入相应的list的dict里；还要放入对应

取出captionid对应的caption，放到list里，
对每个caption，做tokenize，padding，然后存入相应位置

然后取数据计算分数
scores = np.zeros([100,500])
for each img
	copy img to batch_size
	for every 100 caption
		input img and caption
		get 100 scores
		save scores
得到 100 * 500 的数

然后计算rank，应该通过argmax（dims=True）
然后取出每个img对应的5个caption的排名


统计小于1的数量除以全部数量




