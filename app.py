"""HSK 1-3 Sentence Assessor (Streamlit).  Run:  streamlit run app.py

Three features only: (1) Sentence Assessor, (2) searchable vocab table, (3) session progress tracker in the sidebar.
Settings (backend / model / key) come from environment variables or Streamlit secrets:
  LLM_BACKEND = ollama | openrouter     LLM_MODEL = qwen2.5:3b | qwen/qwen-2.5-7b-instruct     OPENROUTER_API_KEY

Display-only safeguards added for the demo (the evaluated pipeline in src/ is unchanged):
  - Pinyin under the correction is generated in code with pypinyin, not taken from the model.
  - If the explanation comes back in Chinese, the app asks once more for English.
  - Grammar rules are shown only when the sentence needed a fix.
"""
import html, json, os, re, sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Streamlit Cloud secrets -> environment, BEFORE src.llm is imported (it reads them at import time)
for _k in ("LLM_BACKEND", "LLM_MODEL", "OPENROUTER_API_KEY"):
    try:
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
    except Exception:
        pass

import jieba                                   # noqa: E402
from pypinyin import pinyin, Style             # noqa: E402
from src import llm, pipeline, vocab_check     # noqa: E402

st.set_page_config(page_title="HSK 1-3 Sentence Assessor", page_icon="📝", layout="centered")

MODEL_CHOICES = {"ollama": ["qwen2.5:3b", "qwen2.5:7b"], "openrouter": ["qwen/qwen-2.5-7b-instruct"]}
EXAMPLES = ["我喜欢喝茶。", "我去商店昨天。", "wo qu shangdian zuotian"]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 960px;}
.hero {background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%); border: 1px solid #bae6fd;
       border-radius: 16px; padding: 1.3rem 1.6rem; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; padding: 0; font-size: 1.8rem; color: #0c4a6e;}
.hero p {margin: .35rem 0 0 0; color: #334155; font-size: 1rem;}
.hanzi {font-size: 2.6rem; font-weight: 600; line-height: 1.35; color: #0c4a6e;}
.pill {display: inline-block; padding: .15rem .75rem; border-radius: 999px; font-size: .85rem; font-weight: 600;
       margin-bottom: .4rem;}
.pill-ok {background: #dcfce7; color: #166534;}
.pill-fix {background: #fef3c7; color: #92400e;}
.small-label {color: #64748b; font-size: .85rem; margin-bottom: 0;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached resources
@st.cache_data
def load_vocab_df():
    df = pd.DataFrame(json.load(open(os.path.join(ROOT, "data", "hsk_vocab.json"), encoding="utf-8")))
    for c, sep in (("meanings", "; "), ("pos", ", ")):          # lists -> plain text, easier to read and search
        if c in df.columns:
            df[c] = df[c].apply(lambda v: sep.join(v) if isinstance(v, list) else v)
    return df


@st.cache_data
def load_rules():
    try:
        rules = json.load(open(os.path.join(ROOT, "data", "hsk_grammar.json"), encoding="utf-8"))["rules"]
        return {r["rule_id"]: r for r in rules}
    except Exception:
        return {}


@st.cache_resource(show_spinner="Loading the grammar index (first run only)...")
def warm_up():
    from src import retrieve
    retrieve._collection()          # builds / opens ChromaDB
    if not retrieve.FAKE_EMBED:
        retrieve._model()           # load the embedding model now, so the first real question is not slow
    return True


VOCAB_DF = load_vocab_df()
RULES = load_rules()
TOTAL_WORDS = len(VOCAB_DF)

# ----------------------------------------------------------------------------- session state
if "practiced" not in st.session_state:
    st.session_state.practiced = set()
if "last" not in st.session_state:
    st.session_state.last = None


# ----------------------------------------------------------------------------- helpers
def words_in(text):
    return {t for t in jieba.lcut(text or "", HMM=False) if t in vocab_check.VOCAB}


def pinyin_of(hanzi):
    """Tone-marked Pinyin made in code from the Hanzi (the model's own Pinyin was often wrong)."""
    words = jieba.lcut(hanzi or "", HMM=False)
    out = " ".join("".join(s[0] for s in pinyin(w, style=Style.TONE)) for w in words)
    return re.sub(r"\s+([。，！？、])", r"\1", out)


_CJK = re.compile(r"[\u4e00-\u9fff]")
_LAT = re.compile(r"[A-Za-z]")


def mostly_chinese(s):
    """True if the text is mainly Chinese characters (quoting a few Chinese words inside English is fine)."""
    s = s or ""
    return len(_CJK.findall(s)) > len(_LAT.findall(s))


def assess_english(text):
    """Run the pipeline; if the explanation comes back in Chinese, ask once more for English."""
    r = pipeline.assess(text, "rag_guardrail")
    if r["status"] == "ok" and (mostly_chinese(r["assessment"]["explanation"]) or mostly_chinese(r["assessment"]["error_type"])):
        old = llm.SYSTEM_PROMPT
        llm.SYSTEM_PROMPT = old + ("\n- IMPORTANT: the explanation and error_type must be written in English words. "
                                   "You may quote Chinese words inside the English sentence.")
        try:
            r2 = pipeline.assess(text, "rag_guardrail")
        finally:
            llm.SYSTEM_PROMPT = old
        if r2["status"] == "ok" and not mostly_chinese(r2["assessment"]["explanation"]):
            r = r2
    return r


def use_example(s):
    st.session_state.sentence = s


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Your progress")
    done = len(st.session_state.practiced & set(VOCAB_DF["hanzi"]))
    st.progress(done / TOTAL_WORDS if TOTAL_WORDS else 0.0)
    st.write(f"**{done} / {TOTAL_WORDS}** HSK 1-3 words practised ({100 * done / TOTAL_WORDS:.1f}%)")
    st.caption("Counts words in the sentences you check this session. Nothing is saved after you close the tab.")
    if st.session_state.practiced:
        with st.expander("Words practised"):
            st.write("  ".join(sorted(st.session_state.practiced)))
    if st.button("Reset progress"):
        st.session_state.practiced = set()
        st.session_state.last = None
        st.rerun()

    st.header("Settings")
    backend = llm.BACKEND
    choices = MODEL_CHOICES.get(backend, [llm.MODEL])
    if llm.MODEL not in choices:
        choices = [llm.MODEL] + choices
    llm.MODEL = st.selectbox("Model", choices, index=choices.index(llm.MODEL))   # shared by all visitors of this demo
    st.caption(f"Backend: {backend}" + (" - local 3B can take 10-20 s per answer." if backend == "ollama" else ""))

# ----------------------------------------------------------------------------- header + tabs
st.markdown("""
<div class="hero">
  <h1>HSK 1-3 Sentence Assessor</h1>
  <p>Write a sentence in Chinese or Pinyin and get friendly feedback, using only beginner grammar.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Practise a sentence", "Vocabulary"])

with tab1:
    st.markdown('<p class="small-label">Try an example</p>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        cols[i].button(ex, key=f"ex{i}", on_click=use_example, args=(ex,))

    with st.form("assess"):
        text = st.text_input("Your sentence (Hanzi, Pinyin, or a mix)", key="sentence",
                             placeholder="我去商店昨天。  or  wo qu shangdian zuotian")
        go = st.form_submit_button("Check my sentence")
    if go:
        if not text.strip():
            st.warning("Please type a sentence first.")
        else:
            try:
                warm_up()
                with st.spinner("Checking..."):
                    st.session_state.last = assess_english(text)
            except Exception as e:                      # e.g. missing API key, no internet, Ollama not running
                st.session_state.last = {"status": "error", "input": text,
                                         "message": f"Sorry, I could not get an answer just now ({type(e).__name__}). "
                                                    "Please try again in a moment."}
            r = st.session_state.last
            if r["status"] == "ok":
                a = r["assessment"]
                st.session_state.practiced |= words_in(r["hanzi_candidate"]) | words_in(a["corrected_hanzi"])
                st.rerun()

    r = st.session_state.last
    if r:
        st.caption(f"You wrote: {r['input']}")
        if r["status"] == "abstained":
            st.info(r["message"])
        elif r["status"] != "ok":
            st.error(r.get("message") or "Something went wrong. Please try again.")
        else:
            a = r["assessment"]
            if r["input_type"] != "hanzi":
                st.caption(f"Read as: {r['hanzi_candidate']}  (automatic guess from your Pinyin)")
            with st.container(border=True):
                if a["is_correct"]:
                    st.markdown('<span class="pill pill-ok">Looks correct</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="pill pill-fix">Needs a fix</span>', unsafe_allow_html=True)
                c1, c2 = st.columns([3, 2])
                c1.markdown('<p class="small-label">' + ("Your sentence" if a["is_correct"] else "Corrected sentence") + '</p>'
                            f'<div class="hanzi">{html.escape(a["corrected_hanzi"])}</div>', unsafe_allow_html=True)
                c1.write(pinyin_of(a["corrected_hanzi"]))
                c2.markdown('<p class="small-label">Meaning</p>', unsafe_allow_html=True)
                c2.write(a["english_translation"])
                st.markdown("**Why**")
                st.write(a["explanation"])
                if mostly_chinese(a["explanation"]):
                    st.caption("This explanation came back in Chinese. Press the button again to try for English.")
                if a.get("error_type") and a["error_type"].lower() != "none" and not mostly_chinese(a["error_type"]):
                    st.caption(f"Type of mistake: {a['error_type']}")
            for w in r["soft_warnings"]:
                st.warning(w["message"])
            if r.get("retrieved") and not a["is_correct"]:
                with st.expander("Grammar rules that may apply"):
                    for x in r["retrieved"]:
                        card = RULES.get(x["rule_id"])
                        if card:
                            st.markdown(f"**{x['rule_id']} - {card['title']}**  \n{card['explanation']}")
                            st.caption(f"Example: {card['correct_example']['zh']} ({card['correct_example']['en']})")
                    st.caption("These are the closest rules found. The answer may not use all of them.")
        st.caption("AI feedback can be wrong, especially for unusual sentences. Check with a teacher for anything important. "
                   "Not for exams or official translation.")

with tab2:
    st.write(f"{TOTAL_WORDS} words, HSK 2.0 Levels 1-3. Words you have practised this session are ticked.")
    f1, f2 = st.columns([3, 1])
    q = f1.text_input("Search (Hanzi, Pinyin or English)")
    df = VOCAB_DF.copy()
    if "level" in df.columns:
        lv = f2.selectbox("Level", ["All"] + sorted(df["level"].astype(str).unique()))
        if lv != "All":
            df = df[df["level"].astype(str) == lv]
    if q.strip():
        mask = df.astype(str).apply(lambda col: col.str.contains(q.strip(), case=False, regex=False)).any(axis=1)
        df = df[mask]
    df.insert(0, "practised", df["hanzi"].isin(st.session_state.practiced).map({True: "✓", False: ""}))
    show = df[[c for c in ["practised", "hanzi", "pinyin", "meanings", "pos", "level"] if c in df.columns]]
    show = show.rename(columns={"practised": "✓", "hanzi": "Word", "pinyin": "Pinyin", "meanings": "Meaning",
                                "pos": "Type", "level": "HSK level"})
    st.dataframe(show, hide_index=True)
    st.caption(f"{len(df)} shown")