![](manual/RecommenderSystems.png)

# RecSys
RecSys 是基于 [RecBole](https://github.com/RUCAIBox/RecBole) ，对推荐系统算法进行实现和评估的学习项目。包含 user_cf、item_cf、NCF/NeuMF、XSimGCL 等模型。

## 目录
- .specify # spec
- manual # 手册
- metric_server # 测评服务器
- RecSys # 推荐系统
  - config # 模型配置文件
  - data # 数据集
  - model # 模型
  - outputs # 输出推荐表
- main.py # 运行入口
- README.md # 说明书


## 启动命令
### 安装依赖，激活虚拟环境
```bash
python setup.py
source .venv/bin/activate
```
### 启动模型
```bash
python main.py --model 模型名 --config 配置文件地址
```
以 dcn 为例
```bash
python main.py --model dcn --config RecSys/config/dcn.yaml
```
### 查看 cuda 实时占用率
```bash
watch -n 1 nvidia-smi
```

## 模型列表
| 模型              |  简介                |   命令                                             |
| :---------------: | :-----------------: | :------------------------------------------------: |
| user_cf           | 基于用户的协同过滤    | --model user_cf --config RecSys/config/user_cf.yaml |
| item_cf           | 基于物品的协同过滤    | --model item_cf --config RecSys/config/item_cf.yaml |
| NCF               |                     | --model ncf --config RecSys/config/ncf.yaml |
| NCF/NeuMF         |                     | --model ncf --config RecSys/config/ncf.yaml |
| XSimGCL           |                     | --model xsimgcl --config RecSys/config/xsimgcl.yaml |



## 推荐算法发展历程
![](manual/推荐系统模型发展历程拓扑图.png)
AI 生成，有待核验
