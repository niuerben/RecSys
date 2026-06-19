![](manual/RecommenderSystems.png)

# RecommenderSystems
本项作为学习用途，对一些推荐系统算法进行实现和评估。包含 user_cf、item_cf、NCF/NeuMF、XSimGCL 等模型。

# 目录
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
```bash
python main.py --model 模型名 --config 配置文件地址
```
以 dcn 为例
```bash
python main.py --model dcn --config RecSys/config/dcn.yaml
```

## 模型列表
user_cf 基于用户的协同过滤
item_cf 基于物品的协同过滤
NCF
NCF/NeuMF
XSimGCL 等模型。

## 推荐算法发展历程
![](manual/推荐系统模型发展历程拓扑图.png)
AI 生成，有待核验
