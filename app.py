import os
import random
import numpy as np
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Contexto",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Minimalist, Professional Styling (Authentic Contexto UI)
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Constrain container width for focused editorial layout */
.block-container {
    max-width: 520px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

/* Header */
.game-header {
    text-align: center;
    padding-bottom: 1rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.2);
}

.game-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}

.game-meta {
    font-size: 0.78rem;
    color: #71717a;
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-top: 0.25rem;
}

/* Minimal Stats Bar */
.stats-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.75rem;
    background: rgba(128, 128, 128, 0.05);
    border: 1px solid rgba(128, 128, 128, 0.12);
    border-radius: 6px;
    font-size: 0.8rem;
    color: #71717a;
    margin-bottom: 1rem;
}

.stats-bar span strong {
    color: inherit;
    font-weight: 600;
}

/* Section Labels */
.section-heading {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #71717a;
    margin-top: 1.25rem;
    margin-bottom: 0.4rem;
}

/* Animations */
@keyframes fillWidth {
    from { width: 0% !important; }
}

@keyframes rowFadeIn {
    from {
        opacity: 0;
        transform: translateY(4px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes spotlightEntrance {
    0% {
        opacity: 0;
        transform: translateY(-6px) scale(0.985);
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes rankPop {
    0% { transform: scale(0.85); }
    60% { transform: scale(1.06); }
    100% { transform: scale(1); }
}

@keyframes alertEntrance {
    0% {
        opacity: 0;
        transform: scale(0.97);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

/* Row Cards matching authentic Contexto */
.guess-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.85rem;
    border-radius: 6px;
    background: rgba(128, 128, 128, 0.04);
    border: 1px solid rgba(128, 128, 128, 0.1);
    margin-bottom: 0.35rem;
    animation: rowFadeIn 0.22s cubic-bezier(0.16, 1, 0.3, 1);
    transition: background-color 0.15s ease, border-color 0.15s ease;
}

.guess-row:hover {
    background: rgba(128, 128, 128, 0.07);
    border-color: rgba(128, 128, 128, 0.18);
}

.guess-row.spotlight {
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.25);
    margin-bottom: 1rem;
    animation: spotlightEntrance 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.guess-row.spotlight .guess-rank {
    animation: rankPop 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}

.guess-word {
    font-size: 0.95rem;
    font-weight: 600;
}

.guess-right {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.guess-rank {
    font-size: 0.85rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    min-width: 3.5rem;
    text-align: right;
}

/* Proportional Color Indicators */
.bar-track {
    width: 90px;
    height: 6px;
    border-radius: 3px;
    background: rgba(128, 128, 128, 0.15);
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 3px;
    animation: fillWidth 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Solid Functional Colors */
.fill-green { background-color: #22c55e; }
.fill-yellow { background-color: #eab308; }
.fill-red { background-color: #ef4444; }

.text-green { color: #22c55e; }
.text-yellow { color: #eab308; }
.text-red { color: #ef4444; }

/* End-game Messages */
.game-alert {
    padding: 0.85rem 1rem;
    border-radius: 6px;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    text-align: center;
    animation: alertEntrance 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}
.alert-win {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #22c55e;
}
.alert-lose {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #ef4444;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Precomputed Dataset (Pure NumPy)
# ---------------------------------------------------------
DATA_FILE = "contexto_data.npz"

@st.cache_resource(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_FILE):
        return None, None, None
    data = np.load(DATA_FILE)
    vocab = data["vocab"].tolist()
    matrix = data["matrix"]
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    return vocab, matrix, vocab_to_idx

vocab, matrix, vocab_to_idx = load_data()

if vocab is None:
    st.error("Data file `contexto_data.npz` not found in workspace.")
    st.stop()

# ---------------------------------------------------------
# Game State Management
# ---------------------------------------------------------
def start_new_game(target=None):
    if target is None:
        candidates = [w for w in vocab[:3000] if len(w) >= 4]
        target = random.choice(candidates)
    else:
        target = target.lower().strip()

    target_idx = vocab_to_idx[target]
    target_vec = matrix[target_idx]

    # Precalculate all cosine similarities via matrix dot product
    sims = matrix @ target_vec
    sorted_indices = np.argsort(-sims)

    rank_lookup = {}
    rank_to_word = {}
    for rank, idx in enumerate(sorted_indices, start=1):
        w = vocab[idx]
        rank_lookup[w] = (rank, float(sims[idx]))
        rank_to_word[rank] = w

    st.session_state.target_word = target
    st.session_state.rank_lookup = rank_lookup
    st.session_state.rank_to_word = rank_to_word
    st.session_state.guess_history = []
    st.session_state.last_guess = None
    st.session_state.is_solved = False
    st.session_state.gave_up = False
    st.session_state.hints_used = 0
    st.session_state.error_message = ""

if "target_word" not in st.session_state:
    start_new_game()

# ---------------------------------------------------------
# Header & Stats Bar
# ---------------------------------------------------------
st.markdown("""
<div class='game-header'>
    <div class='game-title'>Contexto</div>
    <div class='game-meta'>Semantic Word Association • 30,000 Words</div>
</div>
""", unsafe_allow_html=True)

best_rank = f"#{st.session_state.guess_history[0]['rank']:,}" if st.session_state.guess_history else "—"
total_guesses = len(st.session_state.guess_history)
hints_count = st.session_state.hints_used

st.markdown(f"""
<div class='stats-bar'>
    <span>Guesses: <strong>{total_guesses}</strong></span>
    <span>Best: <strong>{best_rank}</strong></span>
    <span>Hints: <strong>{hints_count}</strong></span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Win / Surrender State
# ---------------------------------------------------------
if st.session_state.is_solved:
    st.balloons()
    st.markdown(f"""
    <div class='game-alert alert-win'>
        Solved in <strong>{total_guesses}</strong> guesses. The secret word was <strong>{st.session_state.target_word}</strong>.
    </div>
    """, unsafe_allow_html=True)
elif st.session_state.gave_up:
    st.markdown(f"""
    <div class='game-alert alert-lose'>
        The secret word was <strong>{st.session_state.target_word}</strong> (Rank 1).
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Input Form & Controls
# ---------------------------------------------------------
def process_guess(word_input):
    word = word_input.strip().lower()
    if not word:
        return

    if word not in vocab_to_idx:
        st.session_state.error_message = f"'{word}' is not in the dictionary."
        return

    rank, sim = st.session_state.rank_lookup[word]
    tier = "green" if rank <= 300 else ("yellow" if rank <= 3000 else "red")

    guess_entry = {
        "word": word,
        "rank": rank,
        "similarity": sim,
        "tier": tier
    }
    st.session_state.last_guess = guess_entry
    st.session_state.error_message = ""

    if not any(g["word"] == word for g in st.session_state.guess_history):
        st.session_state.guess_history.append(guess_entry)
        st.session_state.guess_history.sort(key=lambda x: x["rank"])

    if rank == 1:
        st.session_state.is_solved = True

if not st.session_state.is_solved and not st.session_state.gave_up:
    with st.form("guess_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Enter word",
                placeholder="Type a word...",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Enter", use_container_width=True)

        if submitted and user_input:
            process_guess(user_input)
            st.rerun()

if st.session_state.error_message:
    st.caption(f":red[{st.session_state.error_message}]")

# Secondary Controls (Subtle, professional)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Hint", use_container_width=True, disabled=(st.session_state.is_solved or st.session_state.gave_up)):
        closest = st.session_state.guess_history[0]["rank"] if st.session_state.guess_history else 1000
        target_r = max(closest // 2, 2)
        guessed = {g["word"] for g in st.session_state.guess_history}

        for offset in range(len(vocab)):
            for r in [target_r - offset, target_r + offset]:
                if 2 <= r <= len(vocab):
                    cand = st.session_state.rank_to_word[r]
                    if cand not in guessed:
                        process_guess(cand)
                        st.session_state.hints_used += 1
                        st.rerun()
with c2:
    if st.button("Give Up", use_container_width=True, disabled=(st.session_state.is_solved or st.session_state.gave_up)):
        st.session_state.gave_up = True
        st.rerun()
with c3:
    if st.button("New Game", use_container_width=True):
        start_new_game()
        st.rerun()

# ---------------------------------------------------------
# Card Renderer
# ---------------------------------------------------------
def get_pct(rank):
    if rank <= 300:
        return 70 + (300 - rank) / 300 * 30
    elif rank <= 3000:
        return 35 + (3000 - rank) / 2700 * 35
    else:
        return max(6, 35 - (rank - 3000) / 27000 * 29)

def render_row(g, is_spotlight=False):
    word = g["word"]
    rank = g["rank"]
    tier = g["tier"]
    pct = get_pct(rank)
    row_class = "guess-row spotlight" if is_spotlight else "guess-row"

    st.markdown(f"""
    <div class='{row_class}'>
        <span class='guess-word'>{word}</span>
        <div class='guess-right'>
            <div class='bar-track'>
                <div class='bar-fill fill-{tier}' style='width: {pct:.1f}%;'></div>
            </div>
            <span class='guess-rank text-{tier}'>#{rank:,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Guesses List
# ---------------------------------------------------------
if st.session_state.last_guess:
    st.markdown("<div class='section-heading'>Previous Guess</div>", unsafe_allow_html=True)
    render_row(st.session_state.last_guess, is_spotlight=True)

if st.session_state.guess_history:
    st.markdown(f"<div class='section-heading'>Guesses ({len(st.session_state.guess_history)})</div>", unsafe_allow_html=True)
    for g in st.session_state.guess_history:
        render_row(g, is_spotlight=False)
