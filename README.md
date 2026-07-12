# RAG Pipeline on RAG-Instruct

**Natural Language Processing, 2024/25 — Group Assignment**

A Retrieval-Augmented Generation (RAG) pipeline built end-to-end on the [RAG-Instruct](https://huggingface.co/datasets/FreedomIntelligence/RAG-Instruct) dataset: preliminary corpus analysis, dense + LLM-based document retrieval, fine-tuned generative models, automatic evaluation, and a final interactive chatbot with an optional voice interface.

**Team — RAG Against the Machine:** Christian Ferrareis, Niccolò Giallongo, Manuel del Carmen Fernández, Miguel Planas

---

## The dataset

[RAG-Instruct](https://huggingface.co/datasets/FreedomIntelligence/RAG-Instruct) contains ~40K questions, each paired with an answer and **10 candidate documents** (100 words each) retrieved for that question. Answers were generated automatically by GPT-4o, and only some of the 10 documents actually contain the information needed to answer — making document *ranking* a core sub-problem before generation.

Key stats from our analysis:
- ~35K unique words in the filtered question vocabulary, ~69K in answers, ~373K across all documents
- Average question length ≈ 21 words, average answer length ≈ 91 words
- k ≈ 150 clusters gives a reasonable topical grouping of the questions

---

## Pipeline overview

```
Raw dataset
   │
   ▼
1. Preliminary analysis  →  stats, clustering, indexing, Word2Vec
   │
   ▼
2. Retrieval              →  bi-encoder + cross-encoder + LLM-as-judge ranking
   │                          (which of the 10 docs actually answer the question?)
   ▼
3. Generation              →  fine-tune FLAN-T5 / GPT-2 / TinyLlama (LoRA) on top-k docs
   │
   ▼
4. Evaluation              →  RougeL, EM, BERTScore, BLEU, perplexity, faithfulness, relevance
   │
   ▼
5. Final chatbot           →  LangChain pipeline (retrieval → rerank → LLM) + voice I/O
```

---

## Repository structure

| Path | Description |
|---|---|
| `1-preliminary-analysis.ipynb` | Dataset structure, document/vocab length distributions, K-means clustering of questions, an inverted keyword index, and a trained Word2Vec embedding. |
| `Retrieval/2-document-retrieval-embeddings.ipynb` | Embeds questions/documents with a **bi-encoder** (`multi-qa-mpnet-base-cos-v1`) and reranks with a **cross-encoder** (`stsb-distilroberta-base`). |
| `Retrieval/llm-document-ranking.ipynb`, `Retrieval/azure-llm-document-ranking.ipynb` | **LLM-as-judge** document ranking using **Mistral-7B-Instruct** and **DeepSeek-V3**, used to validate/distill the encoder-based ranking. |
| `Model Generation and Evaluation.ipynb` | The consolidated pipeline: builds the top-k dataset, LoRA-fine-tunes **FLAN-T5** / **GPT-2**, and evaluates base vs. fine-tuned models. |
| `5-base-and-ft-generative-model.ipynb`, `5.1-gpt2-generative-model.ipynb`, `5.1-tinyllama-generative-model.ipynb` | Base-vs-fine-tuned comparisons for **FLAN-T5**, **GPT-2**, and **TinyLlama-1.1B-Chat**. |
| `final-langchain.ipynb` | **Final RAG chatbot**: bi-encoder retrieval → cross-encoder rerank → prompt construction → answer generation via LangChain (FLAN-T5 LoRA or Gemini). |
| `TTS_STT.ipynb` | Voice-interface extension: **Whisper** for speech-to-text, **Tacotron2 + WaveGlow** for text-to-speech. |

---

## Methodology

### 1. Preliminary analysis
Corpus statistics (document/vocabulary size and length distributions), K-means clustering of questions to surface topics, an inverted index for keyword search, and a Word2Vec embedding trained on the corpus to inspect word-similarity structure.

### 2. Retrieval — ranking the 10 candidate documents
Since only some of each question's 10 documents are actually relevant, we treat "which documents answer the question" as a ranking problem, tackled three ways:
- **Bi-encoder + cross-encoder**: fast semantic similarity (bi-encoder) followed by pairwise reranking (cross-encoder) for higher-precision top-k selection.
- **LLM-as-judge**: Mistral-7B-Instruct and DeepSeek-V3 rank documents by relevance, used both as an alternative ranking source and as a "teacher" signal to sanity-check the encoder-based ranking.

### 3. Generation — fine-tuning on top-k context
Using the top-k ranked documents (k = 2 or 3) as context, we fine-tune multiple model families with **LoRA** adapters (and, for the smaller models, full fine-tuning) and compare against their base/zero-shot counterparts:
- **FLAN-T5** (small/base) — seq2seq
- **GPT-2** — causal LM
- **TinyLlama-1.1B-Chat** — small instruction-tuned LLM

### 4. Evaluation
Each model is scored on the held-out set with:
- **RougeL**, **BLEU**, **BERTScore F1** — lexical/semantic overlap with the reference answer
- **Exact Match** — strict correctness
- **Perplexity** — fluency
- **Faithfulness** — cosine similarity between the generated answer and the source documents (is the answer grounded?)
- **Relevance** — cosine similarity between the generated answer and the question

Example results (GPT-2, base vs. LoRA fine-tuned):

| Model | RougeL | EM | BERTScore F1 | Perplexity |
|---|---|---|---|---|
| GPT-2 base | 0.1461 | 0.0000 | 0.8354 | 32.45 |
| GPT-2 fine-tuned (LoRA) | 0.1481 | 0.0000 | 0.8372 | 28.94 |
| T5-base | 0.2684 | 0.0000 | 0.8774 | 8.12 |

Fine-tuning consistently improves fluency (lower perplexity) and overlap scores, though exact-match remains near zero — expected, since answers are open-ended generations rather than short extractive spans.

### 5. Final chatbot & extensions
`final-langchain.ipynb` wires the full pipeline together with **LangChain**: a query is embedded and matched against the corpus (bi-encoder), reranked (cross-encoder), and the retrieved documents are inserted into a prompt sent to the generation model (fine-tuned FLAN-T5, or Gemini as a stronger LLM backend). `TTS_STT.ipynb` extends this into a voice-interactive chatbot using Whisper for transcription and Tacotron2/WaveGlow for spoken responses.