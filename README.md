# Pytorch_Train_Demo

> 1. 基本功能：

- 利用accelerate进行分布式训练，混合精度训练等  
  - `accelerate config`
- 利用accelerate实现断点续传 
  - `accelerate.save_state()`
  - `accelerate.load_state()`
  - `accelerate.skip_first_batches()`

- 利用`torch.utils.tensorboard`实现训练可视化
  - 可视化包含loss，accuracy，以及训练过程中模型的参数和梯度分布
  - `accelerate.log()`无法添加histogram图，有局限性
- 灵活的验证策略
  - 由`valid_stategy`和`valid_steps`控制
  - `valid_strategy = "epoch"` && `valid_steps = 1`表示每训练1个epoch验证一次
  - `valid_strategy = "step"` && `valid_steps = 100`表示每训练100个step验证一次

> 2. 打印日志的loss说明：（accuracy同理）

- 第1个epoch，第n个step的loss是前n个step的loss的平均

- 第2个epoch，第n个step的loss是前n个step的loss的平均
- 以此类推......
- 此外，`Tensorboard`中表示loss和accuracy的折线图横坐标均为global_step

> 3. 运行命令：

```python
cd scripts/
accelerate launch --config_file="../configs/accelerate_config.yaml" main.py
```

> 4. 运行环境：

- 我使用`conda env export > my_env.yaml`导出了环境，你可以像这样导入环境：

```bash
conda env create -n YOUR_ENV_NAME -f my_env.yaml
```

> 5. Debug：`.vscode/launch.json`

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
                "--config_file", "/data/home/wupengpeng/code/temp/Pytorch_Train_Demo/configs/accelerate_config.yaml",
                "/data/home/wupengpeng/code/temp/Pytorch_Train_Demo/scripts/main.py",
                "--config", "/data/home/wupengpeng/code/temp/Pytorch_Train_Demo/configs/config.toml",
            ],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

🎉 待补充的功能：动态保存验证集上准确率最高的模型参数（如果是peft_model，则保存adaptor，否则保存整个模型的参数~）
