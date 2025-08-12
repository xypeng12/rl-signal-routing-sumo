# Joint Optimization of Traffic Signal Control and Vehicle Routing using MADRL

This repository contains the implementation of **Multi-Agent Deep Reinforcement Learning (MADRL)** for the **joint optimization of traffic signal control and vehicle routing** in signalized road networks.  


## Overview

- **Signal Agents (SAs)**: Control traffic signal timings at intersections.
- **Routing Agents (RAs)**: Assign vehicle routes dynamically.
- **Shared observations and rewards**: Encourage cooperation between SAs and RAs.
- **Algorithm**: Multi-Agent Advantage Actor-Critic (MA2C) with custom deep neural network architectures.
- **Environment**: Modified Sioux Falls network in SUMO.
- **Objective**: Improve overall network performance compared to controlling signals or routes alone.

---

The approach is based on our paper:

> **Peng, Xianyue**, Han, Gengyue, Wang, Hao, and Zhang, Michael.  
> *Joint optimization of traffic signal control and vehicle routing in signalized road networks using multi-agent deep reinforcement learning.*  
> arXiv preprint arXiv:2310.10856, 2023.  
> [Paper Link](https://arxiv.org/abs/2310.10856)

If you use this code in your research, please cite:
```
@article{peng2023joint,
  title={Joint optimization of traffic signal control and vehicle routing in signalized road networks using multi-agent deep reinforcement learning},
  author={Peng, Xianyue and Han, Gengyue and Wang, Hao and Zhang, Michael},
  journal={arXiv preprint arXiv:2310.10856},
  year={2023}
}
```

---

## Code Reference

Part of the environment setup and MA2C training framework is adapted from:
cts198859 – deeprl_signal_control
GitHub: https://github.com/cts198859/deeprl_signal_control
