<h1 align="center">UniVTAC</h1>

> UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking<br>
> [arXiv](https://arxiv.org/abs/2602.10093) | [PDF](https://arxiv.org/pdf/2602.10093) | [Website](https://univtac.github.io/) | [HuggingFace Dataset](https://huggingface.co/datasets/byml/UniVTAC) | [Modelscope Dataset](https://modelscope.cn/datasets/byml2024/UniVTAC)

**UniVTAC** is a tactile-aware simulation benchmark for robotic manipulation built on top of **NVIDIA Isaac Lab** and **TacEx (UIPC-based tactile simulation)**. It provides a unified framework for collecting expert demonstrations, training visuotactile policies, and evaluating them across a diverse suite of contact-rich manipulation tasks — all with high-fidelity tactile feedback from simulated GelSight Mini, ViTai GF225, or XenseWS sensors.

## Installation

See the [Installation Guide](./docs/Installation.md) for detailed setup instructions, including installing the environment, installing TacEx from the modified local source and setting up cuRobo for motion planning.

## Task Gallery

UniVTAC currently includes the following manipulation tasks, all featuring tactile sensing:

| Task | Module | Description |
|---|---|---|
| **Collect** | `collect` | Collect contact-rich tactile data for pretraining |
| **Lift Bottle** | `lift_bottle` | Grasp and lift a bottle off a surface near a wall |
| **Lift Can** | `lift_can` | Grasp and lift a cylindrical can |
| **Insert HDMI** | `insert_HDMI` | Insert an HDMI connector into a port |
| **Insert Hole** | `insert_hole` | Precision peg-in-hole insertion |
| **Insert Tube** | `insert_tube` | Insert a tube into a fixture |
| **Pull Out Key** | `pull_out_key` | Extract a key from a lock |
| **Put Bottle in Shelf** | `put_bottle_in_shelf` | Place a bottle onto a shelf |
| **Grasp & Classify** | `grasp_classify` | Grasp an object and classify it by tactile feedback |

## Data Collection

See the [Data Collection Guide](./docs/Collection.md) for instructions on how to run the automated data collection pipeline, configure task-specific parameters, and understand the output data structure.

Dataset containing 100 episodes per task can be downloaded from [HuggingFace](https://huggingface.co/datasets/byml/UniVTAC), [Modelscope](https://modelscope.cn/datasets/byml2024/UniVTAC) or by running the script in `data/download.sh`.

## Train & Eval Policies

UniVTAC includes several baseline policies implemented under the `policy/` directory:

- ACT: Action Chunking with Transformers with/without tactile inputs
- Abation: ACT ablation variants for modality comparison
- ViTAL: ACT with CLIP-pretrained tactile-vision encoders in ViTAL

Each policy is a self-contained module under `policy/` with its own data processing, training, and deployment scripts. All policies share a unified evaluation entry point at the project root:

```bash
bash eval_policy.sh ${task_name} ${task_config} ${policy_config} ${gpu_id}
```

For parallel evaluation over many seeds:

```bash
bash parallel_eval.sh ${task_name} ${task_config} ${policy_config} ${gpu_id} [num_processes] [total_num]
```

The evaluation results, including videos and success rate logs, will be saved in the `eval_result/` directory under the project root.

To deploy your own policy, refer to the [Deploy Your Policy](./docs/Deploy.md).

## TODO

- Data collection and evaluation are now only supported on the GelSight Mini sensor. We will add support for ViTai GF225 and XenseWS in the near future.

## 👍 Citations
If you find our work useful, please consider citing:

```
@article{chen2026univtac,
  title={UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking},
  author={Chen, Baijun and Wan, Weijie and Chen, Tianxing and Guo, Xianda and Xu, Congsheng and Qi, Yuanyang and Zhang, Haojie and Wu, Longyan and Xu, Tianling and Li, Zixuan and others},
  journal={arXiv preprint arXiv:2602.10093},
  year={2026}
}
```

## 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

## Contact
<div style="text-align: center;">
  <img src="https://box.nju.edu.cn/seafhttp/f/fc1021a908ff49309f22/?op=view" alt="Wechat Group" width="300"/>
</div>