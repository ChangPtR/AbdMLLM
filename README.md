# AbductiveMLLM: Boosting Visual Abductive Reasoning within MLLMs

<!-- ✨🚀🔧✅📝💡🔍📊📀 💾-->

<p align="center">
    </a>&nbsp&nbsp📖 <a href="https://arxiv.org/abs/2601.02771">ArXiv</a>
    </a>&nbsp&nbsp │ &nbsp&nbsp📖 <a href="doc/appendix.pdf">Appendix</a>
    </a>&nbsp&nbsp │ &nbsp&nbsp🤗 <a href="https://huggingface.co/PtRain/AbductiveMLLM">Models</a>
</p>


## 🔥 News
- [2025/11/11] We release our code on GitHub.
- [2025/11/08] Our work is accepted to AAAI 2026 as Oral presentation 🎉!

## 🔎 Overview
Visual abductive reasoning (VAR) is a challenging task that requires AI systems to infer the most likely explanation for incomplete visual observations. While recent MLLMs develop strong general-purpose multimodal reasoning capabilities, they fall short in abductive inference, as compared to human beings. To bridge this gap, we draw inspiration from
the interplay between verbal and pictorial abduction in human cognition, and propose to strengthen abduction of MLLMs by mimicking such dual-mode behavior. Concretely, we introduce AbductiveMLLM comprising of two synergistic components: REASONER and IMAGINER. The REASONER operates in the verbal domain. It first explores a broad space of possible explanations using a blind LLM and then prunes visually incongruent hypotheses based on cross-modal causal alignment. The remaining hypotheses are introduced into the MLLM as targeted priors, steering its reasoning toward causally coherent explanations. The IMAGINER, on the otherhand, further guides MLLMs by emulating human-like pictorial thinking. It conditions a text-to-image diffusion model on both the input video and the REASONER ’s output embeddings to “imagine” plausible visual scenes that correspond to verbal explanation, thereby enriching MLLMs’ contextual grounding. The two components are trained jointly in an end-to-end manner. Experiments on standard VAR benchmarks show that AbductiveMLLM achieves state-of-the-art performance, consistently outperforming traditional solutions and advanced MLLMs.


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

### Dataset
- For VAR dataset, download from the original repo [VAR](https://github.com/leonnnop/VAR.git).
- For YouCookII dataset, you can download from the official page [URL](http://youcook2.eecs.umich.edu/) or [Huggingface](https://huggingface.co/datasets/lmms-lab/YouCook2). We re-partition the original training and validation sets and adapt YouCookII to the same format as VAR.

The train/val split files for both datasets and other model input files are under `/resources`.

## 📈 Inference & Evaluation

### Inference
We release LoRA adapters fine-tuned on the two datasets on Huggingface. Please first download the weights of base model `Qwen/Qwen2-VL-7B-Instruct` from Huggingface. Then, follow the example on Model Card to merge the adapters to the base model.

Inference code:
```bash
python eval_qwen2vl.py
```  

### Evaluation Procedure
We follow the evaluation procedure in [VAR](https://github.com/leonnnop/VAR.git). Run the command below:
```bash
python -m eval_kit.evaluate_models path/to/your/inference_result.json
```

## 🙏 Acknowledgements
We gratefully acknowledge the contributions of the open-source community, particularly [VAR](https://github.com/leonnnop/VAR.git), [SimDA](https://github.com/ChenHsing/SimDA).

## 📚 Citations

If you find this work helpful, please consider citing:

```
@article{chang2026abductivemllm,
  title={AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs},
  author={Chang, Boyu and Wang, Qi and Guo, Xi and Nan, Zhixiong and Yao, Yazhou and Zhou, Tianfei},
  journal={arXiv preprint arXiv:2601.02771},
  year={2026}
}
```
