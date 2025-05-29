### 1~配置环境
版本：python3.10，cuda12.1
```python
python3 -m venv myenv       # 创建虚拟环境（myenv 是环境名）
source myenv/bin/activate   # 激活虚拟环境（Linux/macOS）

# Windows 下激活：
# myenv\Scripts\activate
pip install -r requirements.txt     # 安装使用的包
```

##### torch版本 ≥ 2.6.0    cuda==‘12.4’，不确定其他版本是否能正常运行，如要安装，可以在myenv环境下输入：

```shell
(venv)root@ssrhhh233#:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 2~训练脚本

```shell
python train.py
```

会在同一目录下生成label_encoder.pkl，和/saved_bert_model文件夹

其中/saved_bert_model文件夹预训练模型在这里[下载](https://drive.google.com/drive/folders/1rFZf9zYH5sfRPxnLhpghcNu3nOdSOWP1?usp=sharing)

### 3~推理脚本

```shell
python inference.py
```

测试样例来源：[“再这么下去，苹果要吃不起了”](https://baijiahao.baidu.com/s?id=1833022327092907978)

准确率just 84%左右，我把它归结于数据集的问题。

### 4~数据集

```json
merge.jsonl
{
	"text":新闻
	"label":0/1 # 表示假/真
	"model":[deepseek-r1, 
			ERNIE-X1-32K,
			gpt-3.5-turbo,
			llama3.3-70b-instruct,
			qwen3-235b-a22b,
			human]
}
```

### 5~训练改进方向

性能资金有限改不了，除非人工制造高质量数据集，目前样本量（6500+条）太少且质量参差不齐，训练效果一般。
