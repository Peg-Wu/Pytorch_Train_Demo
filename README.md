<img src="./logo/logo.png" alt="Pytorch_Train_Demo" style="zoom: 67%;" />

## 1. 基本功能

- 利用accelerate进行分布式训练，混合精度训练等  
  - `accelerate config`
- 利用accelerate实现断点续传 
  - `accelerate.save_state()`
  - `accelerate.load_state()`
  - `accelerate.skip_first_batches()`

- 利用`torch.utils.tensorboard`实现训练可视化
  - 可视化包含loss，accuracy，以及训练过程中模型可训练参数梯度分布
  - `accelerate.log()`：无法添加histogram图，有局限性
- 灵活的验证策略
  - 由`valid_stategy`和`valid_interval`控制
  - `valid_strategy = "epoch"` && `valid_interval = 1`表示每训练1个epoch验证一次
  - `valid_strategy = "global_step"` && `valid_interval = 100`表示每训练100个global_step验证一次
- early_stop策略
  - 模型在`early_stop`个`epoch`后，如果验证集损失没有下降，则提前终止训练
  - 启用early_stop策略时会动态保存验证集上损失最低的模型，保存目录为`./checkpoints/early_stop_best_model`
  - 建议在训练小模型的时候启用early_stop
  - 注意：启用early_stop后不能使用断点续训，如果启用断点续训，则会重新初始化验证集最佳损失
- 保存训练结束模型：`./checkpoints/training_end_model`


## 2. 打印日志

- 第1个epoch，第n个step的loss是前n个step的loss的平均
- 第2个epoch，第n个step的loss是前n个step的loss的平均
- 以此类推...... (accuracy同理)

## 3. 运行命令

```python
# 多卡运行
cd Pytorch_Train_Demo/
accelerate launch --config_file="./accelerate_config.yaml" main.py

# 单卡运行
cd Pytorch_Train_Demo/
CUDA_VISIBLE_DEVICES=0 python main.py
```

## 4. 运行环境

```bash
conda create -n <your_env_name> python==3.9
conda activate <your_env_name>

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install notebook ipywidgets accelerate evaluate scikit-learn peft
```

## 5. Debug

`.vscode/launch.json`

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: accelerate launch",
            "type": "debugpy",
            "request": "launch",
            "module": "accelerate.commands.launch",
            "args": [
                "--config_file", "accelerate_config.yaml",
                "main.py"
            ],
            "console": "integratedTerminal",
            "cwd": "/fs/home/wupengpeng/code/Pytorch_Train_Demo/",
            "justMyCode": true
        }
    ]
}
```

## 6. Contact

- 如果代码中有任何问题，请与我联系：peg2_wu@163.com