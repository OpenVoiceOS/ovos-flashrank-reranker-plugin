[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/ovos-flashrank-reranker-plugin)

# FlashRankReranker OVOS Plugin

This plugin adds FlashRank-based reranking to the Open Voice OS (OVOS) platform. It picks the best answer to a question from a list of candidate answers. It can also extract the most relevant sentence from a text passage. It uses the [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) library to score and rank candidates by relevance to the query.

## Install

```bash
pip install ovos-flashrank-reranker-plugin
```

## Configuration

The `common_query` framework in OVOS uses a `MultipleChoiceSolver` to pick the best answer from several skill responses. Set this plugin as the reranker:

```json
"common_query": {
  "reranker": "ovos-flashrank-reranker-plugin",
  "ignore_skill_scores": true,
  "ovos-flashrank-reranker-plugin": {"model": "ms-marco-TinyBERT-L-2-v2"}
}
```

> NOTE: On a Raspberry Pi, this adds up to 1 second of extra latency to the common query pipeline.

### Available models

The default model is `ms-marco-MultiBERT-L-12`, because it supports many languages.

| Model name | Description |
|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ms-marco-TinyBERT-L-2-v2` | [Model card](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2): trained on the MS MARCO passage ranking task, for machine reading comprehension |
| `ms-marco-MiniLM-L-12-v2` | [Model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2): trained on MS MARCO passage ranking, with fewer documents ranked per second than some other versions but higher accuracy |
| `ms-marco-MultiBERT-L-12` (default) | Multilingual: [supports 100+ languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages) |
| `ce-esci-MiniLM-L12-v2` | [Fine-tuned on the Amazon ESCI dataset](https://github.com/amazon-science/esci-data) for English, Japanese, and Spanish queries: maps text to a 384-dimensional vector space for clustering and product search |
| `rank-T5-flan` | [Model card](https://huggingface.co/bergum/rank-T5-flan): the best-performing non-cross-encoder reranker in this list |
| `rank_zephyr_7b_v1_full` (4-bit-quantized GGUF) | A 7B-parameter GPT-like model fine-tuned on task-specific listwise reranking data |

## Usage

### FlashRankMultipleChoiceSolver

`FlashRankMultipleChoiceSolver` picks the best answer to a question from a list of options. In a retrieval chatbot, a user query returns a list of predefined answers. The solver ranks these options by relevance to the query and returns the most suitable one.

```python
from ovos_flashrank_solver import FlashRankMultipleChoiceSolver

solver = FlashRankMultipleChoiceSolver()
a = solver.rerank("what is the speed of light", [
    "very fast", "10m/s", "the speed of light is C"
])
print(a)
# 2024-07-22 15:03:10.295 - OVOS - __main__:load_corpus:61 - DEBUG - indexed 3 documents
# 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 1 (score: 0.7198746800422668): the speed of light is C
# 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 2 (score: 0.0): 10m/s
# 2024-07-22 15:03:10.297 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 3 (score: 0.0): very fast
# [(0.7198747, 'the speed of light is C'), (0.0, '10m/s'), (0.0, 'very fast')]

# NOTE: select_answer is part of the MultipleChoiceSolver base class, and it uses rerank internally
a = solver.select_answer("what is the speed of light", [
    "very fast", "10m/s", "the speed of light is C"
])
print(a)  # the speed of light is C
```

### FlashRankEvidenceSolverPlugin

`FlashRankEvidenceSolverPlugin` extracts the most relevant sentence from a text passage that answers a given question. It uses the FlashRank algorithm to score and rank sentences by relevance to the query.

```python
from ovos_flashrank_solver import FlashRankEvidenceSolverPlugin

config = {
    "lang": "en-us",
    "min_conf": 0.4,
    "n_answer": 1
}
solver = FlashRankEvidenceSolverPlugin(config)

text = """Mars is the fourth planet from the Sun. It is a dusty, cold, desert world with a very thin atmosphere. 
Mars is also a dynamic planet with seasons, polar ice caps, canyons, extinct volcanoes, and evidence that it was even more active in the past.
Mars is one of the most explored bodies in our solar system, and it's the only planet where we've sent rovers to roam the alien landscape. 
NASA currently has two rovers (Curiosity and Perseverance), one lander (InSight), and one helicopter (Ingenuity) exploring the surface of Mars.
"""
query = "how many rovers are currently exploring Mars"
answer = solver.get_best_passage(evidence=text, question=query)
print("Query:", query)
print("Answer:", answer)
# 2024-07-22 15:05:14.209 - OVOS - __main__:load_corpus:61 - DEBUG - indexed 5 documents
# 2024-07-22 15:05:14.209 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 1 (score: 1.39238703250885): NASA currently has two rovers (Curiosity and Perseverance), one lander (InSight), and one helicopter (Ingenuity) exploring the surface of Mars.
# 2024-07-22 15:05:14.210 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 2 (score: 0.38667747378349304): Mars is one of the most explored bodies in our solar system, and it's the only planet where we've sent rovers to roam the alien landscape.
# 2024-07-22 15:05:14.210 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 3 (score: 0.15732118487358093): Mars is the fourth planet from the Sun.
# 2024-07-22 15:05:14.210 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 4 (score: 0.10177625715732574): Mars is also a dynamic planet with seasons, polar ice caps, canyons, extinct volcanoes, and evidence that it was even more active in the past.
# 2024-07-22 15:05:14.210 - OVOS - __main__:retrieve_from_corpus:70 - DEBUG - Rank 5 (score: 0.0): It is a dusty, cold, desert world with a very thin atmosphere.
# Query: how many rovers are currently exploring Mars
# Answer: NASA currently has two rovers (Curiosity and Perseverance), one lander (InSight), and one helicopter (Ingenuity) exploring the surface of Mars.

```

In this example, `FlashRankEvidenceSolverPlugin` finds and returns the sentence from the text passage that answers the query about the number of rovers exploring Mars.

## Related projects

- [ovos-common-query-pipeline-plugin](https://github.com/TigreGotico/ovos-common-query-pipeline-plugin): the OVOS pipeline that uses `MultipleChoiceSolver` plugins like this one to rank answers from multiple skills.

## Credits

![image](https://github.com/user-attachments/assets/809588a2-32a2-406c-98c0-f88bf7753cb4)

This work was sponsored by VisioLab, part of [Royal Dutch Visio](https://visio.org/). Royal Dutch Visio is a test, education, and research center for innovative assistive technology for blind and visually impaired people and professionals. It explores technological developments in voice, VR, and AI, and shares the resulting knowledge with everyone.
