## Read Me

### Parameters

- **now_method**: This parameter specifies the embedding method that we want to compute, such as `svd`, `sap`, `naive`, etc. You can find the available embeddings at https://han-attention.s3.amazonaws.com/input/drug_side/.
- **now_step**: This parameter specifies the step size that we want to use. For SAPBERT, a suitable step size is `3e-4`. For other methods, we can set the step size to `1e-3` to ensure that the results converge quickly.

### Running Example

To run the script with the specified parameters, you can use the following code:

```R
now_method <- "sap"
now_step <- 1e-3
setwd('/home/ec2-user/SageMaker/drug_side/run')
source("drug_side.r")
```

This code sets the `now_method` parameter to `"sap"` and the `now_step` parameter to `1e-3`. It then changes the working directory to `/home/ec2-user/SageMaker/drug_side/run` and sources the `drug_side.r` script to run it with the specified parameters.