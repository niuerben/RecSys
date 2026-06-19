# spec 开发规范

## 项目结构
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

## 模块
### .specify
从不修改
### manual
从不修改
- references 
本项目参考 https://github.com/RUCAIBox/RecBole 和 https://github.com/RUCAIBox/RecBole-GNN/tree/main
### metric_server
从不修改
### RecSys
根据需要修改
- data
- model
  - 设置早停机制：如果验证指标连续大约 10 次没有提升，就提前停止，输出推荐表，终端输出测试集评估。
  - 输出协议：不每epoch在输出推荐表，只在最终结果输出推荐表。用户推荐表[/RecSys/outputs/modelname_recommendation.csv]，6040行；101列，包含user_id,rec1,rec2,...rec100
  - 模型目录名采用全小写，如[item_cf,user_cf,dcn,ncf,xsimgcl...]
  - 终端输出协议参考：
  ```bash
  使用设备:
  加载 ${model_name} 数据...
  用户数:6040，电影数:3628，交互数:836478，训练:675255，Top100 推荐列: 604000
  加载模型 ${model_name}...
  Epoch 1/500 - loss: 0.4867 - valid_mrr@10: 0.1972 *best*
  Epoch 2/500 - loss: 0.3602 - valid_mrr@10: 0.2022 *best*
  ...
  Epoch 79/500 - loss: 0.2052 - valid_mrr@10: 0.3446 - stale:10/10
  早停触发: valid MRR@10 连续 10 次没有提升，停止于 epoch 79。
  加载验证集最佳 checkpoint: epoch 69, Valid MRR@10=0.3520
  测试集 Test RECALL@10 : 0.1338    MRR@10 : 0.3241    NDCG@10 : 0.1735    HIT@10 : 0.6477    PRECISION@10 : 0.1327
  generating ${model_name} recommendation result: /home/niuerben/1_Learn/RecommenderSystems/RecSys/outputs/${model_name}_recommendation.csv
  ```
- outputs
  - 从不修改
### main.py
根据需要修改
- 运行命令：run main.py [--model 模型] [--config 配置.yaml]，直接把模型解析为小写，然后再/model中寻找相应的模型
### README.md
从不修改