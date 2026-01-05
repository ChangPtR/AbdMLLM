# AbductiveMLLM: Boosting Visual Abductive Reasoning within MLLMs

<!-- ✨🚀🔧✅📝💡🔍📊📀 💾-->


## 🔥 News
- [2025/11/11] We release our code on GitHub.
- [2025/11/08] Our work is accepted to AAAI 2026 as Oral presentation 🎉!

## 🛠️ Set up
### Installation
```bash
git clone https://github.com/ChangPtR/AbdMLLM.git
cd AbdMLLM

conda create -n abdmllm python=3.10
conda activate abdmllm

pip install -r requirements.txt
```
After that, you can install flash-attention from [wheels](https://github.com/Dao-AILab/flash-attention/releases).

<!-- ## 🚀 Training

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
-->

## 📈 Inference & Evaluation

```bash
python eval_qwen2vl.py
```  

### Evaluation Procedure
We follow the evaluation procedure in [VAR](https://github.com/leonnnop/VAR.git). Run the command below:
```bash
python -m eval_kit.evaluate_models path/to/your/inference_result.json
```


