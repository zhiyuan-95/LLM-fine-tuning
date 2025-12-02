# LLM Fine-Tuning with LoRA and Automated Style Evaluation

This project fine-tunes a Gemma-2-9B model using LoRA to generate responses in a specific writing style (e.g., Leprechaun or Yoda). It includes a complete training pipeline, inference functions, and a robust evaluation system that uses another LLM as a “judge” to score stylistic similarity.

---

## 1. Dataset and Formatting

Training data is built using a templated chat format:

```
<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn>
```

Each sample contains:

* instruction (user question)
* response_style (target answer in a chosen style)

This structure allows the base model to learn both content and stylistic patterns.

---

## 2. Tokenization and Model Loading

The project uses:

* HuggingFace `AutoTokenizer`
* Gemma family of models
* Device-mapped loading for GPU/TPU
* Template-based prompt construction for consistent token boundaries

Both generation and manual step-by-step decoding are implemented for debugging and experimentation.

---

## 3. Parameter-Efficient Fine-Tuning (LoRA)

LoRA is applied to reduce training cost while retaining model quality.

Key details:

* Rank r = 8
* Target modules include q_proj, k_proj, v_proj, o_proj, up/down projection layers
* Only ~1–2% of parameters become trainable

Training loop:

* Applies chat template
* Computes cross-entropy loss only over model answer tokens
* Uses Lion optimizer for fast convergence
* Supports monitoring at configurable intervals

Loss curves show consistent decline and steady stylistic drift toward the target style.

---

## 4. Inference (chat API)

A `chat()` function is provided for convenient text generation:

* Builds chat prompt using template
* Encodes and sends to the model
* Supports `only_answer=True` to strip user prompt
* Adjustable temperature and max tokens

This makes the fine-tuned model easy to test interactively.

---

## 5. Evaluation Using an LLM Judge

Automated evaluation is done by another LLM (through OpenRouter) acting as a style-similarity judge.

The judge system prompt gives:

* A reference example
* Style definition
* Instructions to output a score from 0–10
* Strict requirement: return only JSON

The judge receives text such as:

```
Evaluate this text: {generated_text}
```

and returns a score representing stylistic similarity.

---

## 6. The JSON Parsing Problem and the Fix

Real LLMs do not always obey instructions. Instead of returning:

```
{"score": 7}
```

the judge model sometimes returned:

* Multiple JSON blocks
* Extra braces
* Code fences (```json)
* Commentary or explanations
* Trailing garbage

This caused `json.loads()` to fail with `JSONDecodeError`.

### Solution: Robust JSON Extraction

The fix is to extract only the first valid JSON object containing "score" before parsing.

Algorithm:

1. Receive raw model output.
2. Search for the first balanced `{ ... }` block containing the word `"score"`.
3. Extract only that substring.
4. Parse using `json.loads()`.
5. Normalize score to 0–1.

This makes the scoring pipeline stable, reliable, and tolerant to noisy LLM outputs.

As a result:

* No more parsing crashes
* Evaluation becomes repeatable
* Works with any judge model (Gemma, LFM-40B, Qwen, Claude, etc.)

---

## 7. Style Score Analysis

Using the robust scoring system, the pipeline computes:

* Base style scores (original model)
* Generated scores (fine-tuned model)
* Training-set scores (ground truth style samples)

Scores are analyzed by:

* Mean ± standard deviation
* Histogram comparison (Seaborn)
* Log-likelihood evaluation on held-out tests

This allows objective measurement of how well the model learned the target style.

---

## 8. Final Result

The project produces:

* A fine-tuned Gemma model capable of speaking in a chosen style
* A reusable LoRA training script
* A chat inference function
* A reliable LLM-as-a-judge scoring system
* A robust JSON extractor that prevents evaluation crashes
* Visualizations and metrics for quality assessment

This pipeline can be easily adapted for:

* Role-playing models
* Creative style generation (poetry, dialects)
* Safety alignment
* Instruction-following improvements
* Persona-based agents


