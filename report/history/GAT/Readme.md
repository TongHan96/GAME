# GAT Training Script

This script trains a Graph Attention Network (GAT) using the specified parameters.

## Parameters

- `--ATTENTION_TYPE`: Attention type for GAT. Choices: 'Naive' or 'A'. Default: 'Naive'.
- `--want_TOP1`: Top1 performance threshold for storing model and embedding. Default: 76.2.
- `--drop_out`: Parameter drop_out prob. Default: 0.0.
- `--has_origin_model`: Whether loading original model or not. Default: False.
- `--lr`: Parameter learning rate. Default: 5e-4.
- `--AA`: Parameter AA. Default: 1.
- `--BB`: Parameter BB. Default: 5.
- `--lambd`: Parameter lambd. Default: 0.5.
- `--scale_one_one`: Parameter scale_one_one. Default: 1.
- `--scale_hie`: Parameter scale_hie. Default: 1.
- `--res`: Whether use resnet not not. Default: True.
- `--scale_OTOL`: Parameter scale_one_one. Default: 50.
- `--low_dim`: Parameter low_dim. Default: 0.
- `--Truncate`: Parameter Truncate. Default: 0.
- `--path_origin`: Train from the initial model and embedding path_origin is not None. Default: None.
- `--FROZEN`: Whether the original model be trained not not. Default: False.
- `--epochs`: Total Epochs. Default: 3.
- `--CHECK_ALL`: whether to check attention or not. Default: False.
- `--latent`: whether to generate latent nodes or not. Default: False.

## Examples

To run the script with the default parameters, you can use the following command:

```
%run main.py
```

To specify custom parameters, you can use the following command:

```
%run main.py --ATTENTION_TYPE “naive” --want_TOP1 72.0 --epochs 3 --AA 1 --BB 1 --lambd 0.5 --drop_out 0.1 --scale_one_one 10 --res True %run main.py --ATTENTION_TYPE “A” --want_TOP1 72.0 --epochs 5 --AA 1 --BB 3 --lambd 0.5 --drop_out 0.1 --low_dim 1 --lr 3e-4 --res True
```


