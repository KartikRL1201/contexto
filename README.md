# Contexto AI — Word Association Game

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://contexto-2tea22ftz55mnavwv4xm6m.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized%20Math-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered word association game inspired by **Contexto**, powered by dense geometric word embeddings (**GloVe 100d**) and vectorized linear algebra. Players guess a secret target word, guided by real-time semantic similarity rankings across a curated 30,000-word English vocabulary.

🎮 **[Play Live Demo](https://contexto-2tea22ftz55mnavwv4xm6m.streamlit.app/)**

---

## 📌 Overview

Traditional word games rely on spelling (e.g., Wordle). **Contexto** operates purely on **linguistic semantics**. 

Words that frequently appear in similar contexts across massive text corpora share spatial proximity in high-dimensional vector space. In this implementation:
- The secret word is mapped to a vector $\mathbf{v}_{\text{secret}} \in \mathbb{R}^{100}$.
- Every guess is evaluated against a pre-normalized vocabulary matrix $X \in \mathbb{R}^{30000 \times 100}$.
- Cosine similarities are computed in **$\approx 2$ milliseconds** using BLAS matrix operations, sorting words from **Rank 1** (the target) to **Rank 30,000** (most distant).

---

## 🧠 Machine Learning & Architecture

### 1. The Vector Space Model
We utilize **GloVe (Global Vectors for Word Representation)** trained on 6 Billion tokens of Wikipedia and Gigaword. Unlike local n-gram window models that emphasize direct synonyms, GloVe captures global co-occurrence statistics, making it ideal for thematic, associative clues (e.g., `coffee` $\leftrightarrow$ `morning`, `caffeine`, `cup`).

### 2. High-Performance Vectorized Cosine Similarity
Cosine similarity between two vectors $\mathbf{u}$ and $\mathbf{v}$ is defined as:

$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \cdot \|\mathbf{v}\|_2}$$

To eliminate runtime overhead, the vocabulary matrix $X$ is pre-normalized offline:

$$\|X_i\|_2 = 1 \quad \forall \, i \in \{1, \dots, 30000\}$$

When a secret word $\mathbf{s}$ is selected, its vector is also normalized ($\|\mathbf{s}\|_2 = 1$). The cosine similarity for all 30,000 words simplifies to a single matrix-vector dot product:

$$\mathbf{sims} = X_{\text{norm}} \cdot \mathbf{s}_{\text{norm}}$$

Ranking is performed via NumPy's C-accelerated `np.argsort(-sims)`, providing instantaneous $O(1)$ lookups during gameplay.

### 3. Decoupled Production Serialization
Rather than shipping heavy NLP frameworks (`gensim`, raw 128 MB embedding text files) to production, the curated 30,000-word vocabulary and pre-normalized float32 embedding matrix are serialized into an **11 MB compressed binary** (`contexto_data.npz`).

- **Zero runtime downloads**: No external model downloading during deployment.
- **Microsecond cold starts**: The production web app loads in $< 100$ ms using pure NumPy.
- **Cross-platform**: Fully compatible with Python 3.10 through Python 3.14+ without C-compiler build steps.

---

## ✨ Features

- **Minimalist, Authentic UI**: Focused editorial layout inspired by the original Contexto game.
- **Previous Guess Spotlight**: Displays the most recent guess right below the input with its rank and proximity bar, preventing it from getting lost in the sorted history.
- **Rank Tiers**:
  - 🟢 **Green (Rank 1 – 300)**: Burning hot / closely related.
  - 🟡 **Yellow (Rank 301 – 3,000)**: Warm / in the same topical ballpark.
  - 🔴 **Red (Rank 3,001+)**: Cold / distant.
- **Adaptive Hint System**: Dynamically searches outward to find an unguessed clue that cuts your current best rank in half, helping you narrow down the semantic direction without spoiling the answer.
- **60fps Micro-Interactions**: Smooth cubic-bezier bar-fill transitions and subtle micro-animations.

---

## 📂 Repository Structure

```
contexto/
├── app.py                         # Production Streamlit Web Application
├── contexto_data.npz              # Serialized 30k vocabulary & pre-normalized matrix (~11 MB)
├── contexto-implementation.ipynb  # Step-by-step Jupyter Notebook for ML exploration
├── requirements.txt               # Production dependencies (pure NumPy + Streamlit)
└── README.md                      # Project documentation
```

---

## 🚀 Local Installation & Setup

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/KartikRL1201/contexto.git
cd contexto
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```
Or:
```bash
python3 -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Machine Learning Notebook

If you wish to explore the step-by-step data science process (loading raw GloVe vectors, vector arithmetic, vocabulary curation, and rank distribution analysis), open:

```bash
jupyter notebook contexto-implementation.ipynb
```

The notebook covers:
1. Environment setup & dependency installation.
2. Vector exploration (e.g. $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$).
3. Curating common English words via `wordfreq`.
4. Matrix pre-normalization and BLAS performance benchmarks.
5. In-notebook interactive CLI and widget gameplay.
6. Exporting to the optimized `.npz` binary.

---

## 🛠️ Tech Stack

- **Frontend / Application**: [Streamlit](https://streamlit.io/)
- **Core Math / Array Operations**: [NumPy](https://numpy.org/)
- **Embeddings & NLP Exploration**: [GloVe](https://nlp.stanford.edu/projects/glove/) via [Gensim](https://radimrehurek.com/gensim/), [Wordfreq](https://github.com/rspeer/wordfreq)
- **Deployment**: Streamlit Community Cloud

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
