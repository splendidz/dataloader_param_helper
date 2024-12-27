# ⚡ DataLoader Parameter Optimization Package

## 🚀 Optimize `torch.utils.data.DataLoader` Parameters Effortlessly

<div style="text-align: justify;">
Finding the optimal configuration for <code>torch.utils.data.DataLoader</code> parameters such as <code>num_workers</code>, <code>prefetch_factor</code>, <code>pin_memory</code>, and <code>persistent_workers</code> can be challenging and system-dependent. This package leverages efficient algorithms to streamline the process, saving you time and effort while enhancing your model's training speed.
</div>

# Overview

This package is designed to optimize the loading parameters for 
`torch.utils.data.DataLoader`, specifically `num_workers`, `prefetch_factor`, 
`pin_memory`, and `persistent_workers`.

These parameters can have varying optimal values depending on the specific hardware 
and system configurations.

<div style="background-color:#f9f9f9; padding:10px; border:1px solid #ddd;">
Manually testing all possible combinations to determine the fastest configuration is 
often an extremely time-consuming and labor-intensive process. This package simplifies 
the task by employing advanced techniques such as <strong>binary search</strong>, 
<strong>early termination</strong>, and <strong>time prediction algorithms</strong> to 
identify the most efficient parameters with minimal testing.
</div>

While the identified configuration may not always guarantee absolute optimality, it is 
designed to outperform default settings and significantly accelerate your training 
pipeline, ensuring your model runs more efficiently.

## Caching and Loading Optimized Parameters

<div style="text-align: justify;">
The process of identifying the optimal parameters may take some time to complete, as it involves thorough testing and analysis. However, this package includes a <strong>built-in caching and loading mechanism</strong> that automatically saves the results after the first test run. In subsequent runs, the saved parameters are loaded automatically, significantly reducing the time required for repeated executions.
</div>

<div style="text-align: justify;">
If desired, this caching feature can be disabled for manual parameter testing. Additionally, after running the tests, you can inspect the saved results file to review the optimized parameters and their corresponding configurations.
</div>

## Installation

<div style="text-align: justify;">
You can install this package directly from PyPI using <code>pip</code>. Follow the command below to install:
</div>

```bash
pip install dataloader-param-helper