# Training on Market1501
## mode=train
- bounding_box_test文件夹 图片命名方式
  - ![](imgs/图片命名方式.png)
  - people_id每个人的编号，同一个人在不同图像上的people_id一样
- [train_market1501](../train_market1501.py).read_train()
  - 获取所有数据的people_id
  - create_validation_split(people_id)
    - 对所有people_id**去重**
    - **去重后的people_id**按一定比例随机采样作为验证集，其余为训练集
    - 返回对应的索引
  - 返回训练集的people_id
- train_x:图片
- train_y:people_id
- [train_app](../train_app.py).create_trainer()
  - /cpu:0
  - ![](imgs/预处理.png)
  - image_var流向CNN网络
  - label_var、feature、logits流向loss

- CNN网络
  - ![](imgs/CNN网络.png)
  - 激活函数都是ELU
- 残差网络
  - ![](imgs/resdual_net.png)
- loss
  - cross_entropy_var(本论文的损失)
    - 计算"预处理流出的label_var"和"CNN网络流出的logits"的交叉熵
  - accuracy_var
    - "logits最大概率的类"和"label_var"计算得到精确度
  - magnet_loss(不是本论文的损失)
    - 计算"预处理流出的label_var"和"CNN网络流出的feature"的自定义损失
  - triplet_loss(不是本论文的损失)
    - 计算"预处理流出的label_var"和"CNN网络流出的feature"的自定义损失

## eval
- 获取验证集
  - 由于[train_market1501](../train_market1501.py).read_validation和read_train的随机种子一样，虽然train和eval不在同一个进程里，但得到的validation和train是刚好分割开的。
- create_network_factory里只有is_training参数不同


# code tips
- def arg_scope(list_ops_or_scope, **kwargs):
  - 向参数添加默认参数
  - Eg:[factory_fn](../nets/deep_sort/network_definition.py)
- 多进程且进程安全地train
  - [queued_trainer](../queued_trainer.py).run()
  - [ ] 生成器生成训练样本
- vscode lanch.json单个进程的环境变量修改
  - command
    - ```
      CUDA_VISIBLE_DEVICES="" python train_market1501.py \
          --mode=eval \
          --dataset_dir=/home/staillyd/DataSets/Market1501/Market-1501-v15.09.15/ \
          --loss_mode=cosine-softmax \
          --log_dir=./output/market1501/ \
          --run_id=cosine-softmax \
          --eval_log_dir=./eval_output/market1501
      ```
  - launch.json
    - ```
      {
          "name": "CMC evaluation on validation set",
          "type": "python",
          "request": "launch",
          "program": "${workspaceFolder}/train_market1501.py",
          "console": "integratedTerminal",
          "args": [
              "--mode=eval",
              "--dataset_dir=/home/staillyd/DataSets/Market1501/Market-1501-v15.09.15/",
              "--loss_mode=cosine-softmax",
              "--log_dir=./output/market1501/",
              "--run_id=cosine-softmax",
              "--eval_log_dir=./eval_output/market1501"
          ],
          "env":{"CUDA_VISIBLE_DEVICES":""}
      }
      ```