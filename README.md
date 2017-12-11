# article_image
首先完成mcnn的结果复现

安装pycocotools
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install

安装必要的包
pip install -r requirements.txt

下载数据文件
chmod +x download.sh
./download.sh

预处理数据
python build_vocab.py
python resize.py