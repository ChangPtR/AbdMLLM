# AbductiveMLLM: Boosting Visual Abductive Reasoning within MLLMs

<!-- ✨🚀🔧✅📝💡🔍📊📀 -->

## 🔥 News
- [2025/11/11] We release our code on GitHub.
- [2025/11/08] Our work is accepted to AAAI 2026 as Oral presentation.

<!-- ## 🛠️ Set up -->
<!-- ### Installation
```bash
git clone https://github.com/QiWang98/VideoRFT
cd VideoRFT

# Create and activate environment
conda create -n VideoRFT python=3.11 
conda activate VideoRFT
bash setup.sh

# Install decord for improved video processing
cd src/qwen-vl-utils
pip install -e .[decord]
```

## 🚀 Training

### Supervised Fine-Tuning (SFT)
We begin with supervised fine-tuning on the VideoRFT-CoT dataset for one epoch:

```bash
bash ./src/scripts/run_sft_video.sh
```

This step can be skipped by directly using our pretrained SFT models, available at [🤗VideoRFT-SFT-7B](https://huggingface.co/QiWang98/VideoRFT-SFT) or [🤗VideoRFT-SFT-3B](https://huggingface.co/QiWang98/VideoRFT-SFT-3B).

### Reinforcement Learning (RL)

Next, perform reinforcement learning using the VideoRFT-RL dataset:

```bash
bash ./src/scripts/run_grpo_video.sh
```

To enable faster training via vLLM acceleration:

```bash
bash ./src/scripts/run_grpo_vllm_qwen25vl.sh
```

> **Note:** During training, we adopt the following settings for efficiency:

* **VIDEO PIXELS**: 128 × 28 × 28
* **FPS FRAMES**: 16

All frame-related configurations can be adjusted in `src/qwen-vl-utils`.

## 📈 Inference & Evaluation

> During inference, we increase the maximum frame resolution and length to boost performance:

* **VIDEO PIXELS**: 256 × 28 × 28
* **FPS FRAMES**: 32

You can configure these parameters in `src/qwen-vl-utils`.

> We evaluate all models under a unified decoding configuration following the official Qwen2.5-VL demo:

* `top_p = 0.001`
* `temperature = 0.01`

### Evaluation Procedure

1. Download preprocessed evaluation JSONs from: \[[🤗 eval](https://huggingface.co/datasets/Video-R1/Video-R1-eval)]

2. Download the video data from the official sites of each benchmark and organize them as specified in the JSON files.

3. Run the evaluation across all benchmarks:

```bash
bash ./src/eval_bench.sh
``` -->

