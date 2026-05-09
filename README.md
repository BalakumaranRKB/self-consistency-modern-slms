# Self-Consistency for Modern Small Language Models
 
Does self-consistency (Wang et al., 2023) still help today's smaller, more efficient language models? This project finds out.
 
## What This Project Does
 
Self-consistency is a simple idea: instead of asking a language model for one answer, you ask it the same question many times (with some randomness), and then pick the answer that comes up most often — basically, majority voting over multiple reasoning attempts.
 
The original paper tested this on massive 137B–540B parameter models. But modern small models (2B–9B parameters) are trained much better than those old giants. **So the question is: do these modern small models still benefit from self-consistency, or are they already good enough on their own?**
 
## Models Tested
 
| Model | Parameters |
|---|---|
| Gemma-2B-it | 2.6B |
| Phi-3-mini-4k-instruct | 3.8B |
| Mistral-7B-Instruct-v0.3 | 7.2B |
| Gemma-2-9B-it | 9B |
 
## How It Works
 
1. Each model is given 8 worked math examples (few-shot chain-of-thought prompting)
2. **Greedy baseline**: the model answers each question once (temperature=0)
3. **Self-consistency**: the model answers each question 40 times (temperature=0.7), and the most common answer wins
4. Both are evaluated on the full GSM8K test set (1,319 grade-school math problems)
## Key Results
 
| Model | Greedy Accuracy | Self-Consistency Accuracy | Gain |
|---|---|---|---|
| Gemma-2B | 50.2% | 59.5% | +9.3% |
| Phi-3-mini | 84.8% | 91.3% | +6.4% |
| Mistral-7B | 53.9% | 72.9% | +19.0% |
| Gemma-9B | 86.2% | 89.5% | +3.3% |
 
**Self-consistency helps all four models**, but how much it helps depends on two things:
- **Headroom** — how much room for improvement exists (100% minus greedy accuracy)
- **Base capability** — the model needs to be good enough to sometimes produce correct answers among its samples
Mistral-7B hit the sweet spot: mediocre greedy accuracy but strong enough reasoning to recover via majority voting.
 
## Dataset
 
[GSM8K](https://huggingface.co/datasets/openai/gsm8k) — 1,319 grade-school math word problems (Cobbe et al., 2021)
 
## Setup & Requirements
 
- Python 3.x
- [vLLM](https://github.com/vllm-project/vllm) for fast batched inference
- An NVIDIA GPU (experiments were run on an A100 40GB via Google Colab Pro)
```bash
pip install vllm ray
```
 
## Files
 
- `SelfConsistency_Code.ipynb` — all experiment code (greedy + self-consistency for all 4 models)
- `Final_Report.docx` — full write-up with analysis, ablation studies, and plots
## References
 
- Wang et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Cobbe et al. (2021). *Training Verifiers to Solve Math Word Problems.* [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
