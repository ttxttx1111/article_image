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



