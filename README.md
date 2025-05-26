# TextualVerifier - Verify Step by Step in TextGrad

## Introduction
1. I want to verify calculation, basically calculation is a result from instance + prompt
2. So, I want to verify calculation only (add & revise if anything need to add/wrong) in calculation only, no need to find the solution, just focus on calculation (based on instance + prompt)
3. The idea is: instance, prompt, calculation going to verify function
4. Then need to CoT prompt (instance) & step formatter (instance)
5. After formatted, I want to each step iteratively as step_eval_iterations running new function to get new variant of calculation, so the input are (instance + prompt) the result is variant of calculation (total as step_eval_iterations)
6. Then, in every end of iterations in step, we vote on the most significant calculation, and save it to use later
7. After all step iterated, then we merge all of vote result in every step into one
8. Then we verify the early calculation from verify function to this one
9. The decision are: 3 classification
10. If the early calculation is not correct -> then update and revise it
11. If the early calculation is correct and some of variant calculation still not in the early calculation -> then update it
12. If the early calculcation is correct and no variant calculation needed to merge -> pass it

## Main References
| Type | Title | Link |
| ---- | ---- | ---- |
| Article | "TextGrad: AutoGrad for Text" by Stanford HAI | https://hai.stanford.edu/news/textgrad-autograd-text |
| Paper | Yuksekgonul, M., Bianchi, F., Boen, J., Liu, S., Huang, Z., Guestrin, C., & Zou, J. (2024). TextGrad: Automatic "differentiation" via text. arXiv. | https://arxiv.org/abs/2406.07496 |
| Documentation | TextGrad's Documentation | https://textgrad.readthedocs.io/en/latest/index.html |
| Github | textgrad | https://github.com/zou-group/textgrad |

## Datasets Sources
| Dataset | Link |
| ---- | ---- |
| BBH Object Counting | https://github.com/suzgunmirac/BIG-Bench-Hard |
| GPQA | https://huggingface.co/datasets/Idavidrein/gpqa |
| GSM8K | https://huggingface.co/datasets/openai/gsm8k |
| Leetcode Hard | https://raw.githubusercontent.com/vinid/data/master/leetcode_with_tests.jsonl |
| MMLU | https://huggingface.co/datasets/cais/mmlu |

## Helpful Arcticles
- https://codeslord.github.io/general/2024/06/12/textgrad/

## License
This project is licensed under the [MIT License](LICENSE).