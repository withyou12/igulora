

## Model Download
Download the corresponding model from the Magic Tower community
`pip install modelscope`
For example, download the llama2-7 model to the `models/Llama2_7B` folder
`modelscope download --model LLM-Research/llama-2-7b --local_dir ./models/Llama2_7B`

## Dataset Download
In the [ModelScope community](https://www.modelscope.cn) or [Huggingface community](https://huggingface.co), you can load corresponding datasets.

## Fine-tuning Steps:
- Stage 1: Perform a pruning training once by running the corresponding `*_prune.sh` script file in the script folder.
- Stage 2: After pruning is complete, perform additional training by running the corresponding `*_final.sh` script file in the script folder.


## Related Papers
- Title: IGU-LoRA: Adaptive Rank Allocation via Integrated Gradients and Uncertainty-Aware Scoring
- Bib style: 
``` 
@inproceedings{Cui2026igulora,
  title={IGU-LoRA: Adaptive Rank Allocation via Integrated Gradients and Uncertainty-Aware Scoring}
  author={Xuan Cui and Huiyue Li and Run Zeng and Yunfei Zhao and Jinrui Qian and Wei Duan and Bo Liu and Zhanpeng Zhou},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}, 
  url={https://openreview.net/pdf?id=MnToYQx9My}
}
```
**Welcome to cite our work**
