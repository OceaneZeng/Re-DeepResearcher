import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 从 .env 文件加载环境变量

# 初始化客户端
client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# -----------------
# Config
# -----------------
INPUT_PARQUET = Path("../data/MMLU-Pro/test-00000-of-00001.parquet")
OUTPUT_PARQUET = Path("../data/MMLU-Pro/synthetic_cot_1000.parquet")

SAMPLE_N = 14000
RANDOM_SEED = 42

BATCH_SIZE = int(os.getenv("COT_BATCH_SIZE", "25"))
SAVE_EVERY_BATCH = True  # 每个 batch 结束就落盘，最稳

MAX_RETRIES = int(os.getenv("COT_MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("COT_RETRY_BACKOFF", "2"))

# 1. 加载测试数据集
df_test = pd.read_parquet(INPUT_PARQUET)

# 随机提取 SAMPLE_N 条数据
if len(df_test) > SAMPLE_N:
    df_sample = df_test.sample(n=SAMPLE_N, random_state=RANDOM_SEED)
else:
    df_sample = df_test

# 2. 定义推理风格模板
FEW_SHOT_PROMPT = """
You are an expert academic assistant. Produce a single-paragraph chain-of-thought in the EXACT format.

Format requirements (MUST follow):
1) Output MUST be English only (no Chinese characters).
2) Output MUST be a single paragraph (no bullet points).
3) Output MUST start with: "A: Let's think step by step." (exactly).
4) Output MUST end with: "The answer is (X)." where X is one of A,B,C,D,E,F,G,H,I,J.
5) Do NOT add extra lines like "Final answer:" outside the paragraph.

Example output:
A: Let's think step by step. A characteristic of a ring is R is $n$ if the statement $ka = 0$ for all $a\in 2Z$ implies that $k$ is a multiple of $n$. Assume that $ka = 0$ for all $a\in 2Z$ for some $k$. In particular $2k = 0$. Hence $k=0$ and $n=0$. The answer is (A).
"""


def _build_prompt(row: pd.Series) -> str:
    return (
        f"{FEW_SHOT_PROMPT}\n"
        f"Question: {row['question']}\n"
        f"Options: {row['options']}\n"
        f"Correct answer: {row['answer']}\n"
        "Write the output in the exact format (one paragraph):"
    )


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _extract_choice_letter(text: str) -> str | None:
    # Try to find a final (X) choice anywhere.
    import re

    m = re.search(r"\(([A-J])\)", text)
    if m:
        return m.group(1)
    # Sometimes models output 'Answer: A'
    m = re.search(r"\b([A-J])\b", text.strip().split()[-1])
    if m:
        return m.group(1)
    return None


def _clean_cot(text: str, *, forced_choice: str | None = None) -> str:
    """Normalize output to the required single-paragraph format."""
    if text is None:
        return ""

    t = str(text).strip()

    # Strip common wrappers
    t = t.replace("```", "").strip()

    # Drop leading labels if present
    for p in ("Reasoning:", "Answer:", "Final Answer:", "Final answer:", "Output:"):
        if t.startswith(p):
            t = t[len(p):].lstrip()

    # Flatten to one paragraph
    t = " ".join(ln.strip() for ln in t.splitlines() if ln.strip()).strip()

    if _contains_cjk(t):
        raise ValueError("Model output contains Chinese characters; enforce English-only.")

    # Ensure required prefix
    prefix = "A: Let's think step by step."
    if not t.startswith(prefix):
        # If the model started without 'A:' but has the phrase, normalize
        if "Let's think step by step." in t:
            # remove any text before the phrase
            idx = t.find("Let's think step by step.")
            t = prefix + " " + t[idx + len("Let's think step by step."):].lstrip()
        else:
            t = prefix + " " + t

    # Decide final choice
    choice = forced_choice or _extract_choice_letter(t)
    if choice is None:
        # If we can't parse, keep as-is but still enforce a trailing answer placeholder.
        choice = "A"

    # Remove any existing trailing answer sentence to avoid duplicates
    import re

    t = re.sub(r"\s*The answer is\s*\([A-J]\)\.?\s*$", "", t).strip()
    t = re.sub(r"\s*Answer\s*:\s*\(?[A-J]\)?\.?\s*$", "", t).strip()

    # Append normalized ending
    t = f"{t} The answer is ({choice})."

    return t


def generate_cot(row: pd.Series) -> str:
    prompt = _build_prompt(row)

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="qwen3.5-397b-a17b",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content

            # Use ground-truth letter (from dataset) if it looks like a single A-J letter.
            forced_choice = None
            ans = str(row.get("answer", "")).strip()
            if len(ans) == 1 and ans.upper() in list("ABCDEFGHIJ"):
                forced_choice = ans.upper()

            return _clean_cot(raw, forced_choice=forced_choice)
        except Exception as e:
            last_err = e
            sleep_s = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"[WARN] request failed (attempt {attempt}/{MAX_RETRIES}): {e}; sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries") from last_err


# -----------------
# Resume support
# -----------------
if OUTPUT_PARQUET.exists():
    print(f"检测到已存在输出文件，尝试断点续跑：{OUTPUT_PARQUET}")
    df_out = pd.read_parquet(OUTPUT_PARQUET)

    # 用 index 对齐（最简单稳妥）。如果你不想依赖 index，也可以改成用某个唯一 key。
    df_sample = df_sample.copy()
    if "cot_content" not in df_sample.columns:
        df_sample["cot_content"] = pd.NA

    if "cot_content" in df_out.columns:
        # 只回填已有的 cot_content
        df_sample.loc[df_out.index.intersection(df_sample.index), "cot_content"] = df_out.loc[
            df_out.index.intersection(df_sample.index), "cot_content"
        ]
else:
    df_sample = df_sample.copy()
    if "cot_content" not in df_sample.columns:
        df_sample["cot_content"] = pd.NA


def _is_done(v) -> bool:
    return v is not None and v is not pd.NA and str(v).strip() != ""


pending_idx = [i for i, v in df_sample["cot_content"].items() if not _is_done(v)]
print(f"总样本: {len(df_sample)}; 待生成: {len(pending_idx)}; batch_size={BATCH_SIZE}")

# 3. 分批生成（可中断可恢复）
print("正在分批生成思维链内容...")

for start in range(0, len(pending_idx), BATCH_SIZE):
    batch = pending_idx[start : start + BATCH_SIZE]
    batch_no = start // BATCH_SIZE + 1
    total_batches = (len(pending_idx) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n=== Batch {batch_no}/{total_batches} ({len(batch)} rows) ===")

    for j, idx in enumerate(batch, start=1):
        row = df_sample.loc[idx]
        try:
            cot = generate_cot(row)
            df_sample.at[idx, "cot_content"] = cot
            print(f"[{batch_no}/{total_batches}] {j}/{len(batch)} idx={idx} OK")
        except Exception as e:
            # 失败也不要中断整个任务：记录错误，继续下一条
            df_sample.at[idx, "cot_content"] = f"__ERROR__: {type(e).__name__}: {e}"
            print(f"[{batch_no}/{total_batches}] {j}/{len(batch)} idx={idx} FAIL: {e}")

    # 4. 保存（每个 batch 落盘，避免中断损失）
    if SAVE_EVERY_BATCH:
        tmp_path = OUTPUT_PARQUET.with_suffix(".parquet.tmp")
        df_sample.to_parquet(tmp_path, engine="pyarrow", index=True)
        tmp_path.replace(OUTPUT_PARQUET)
        print(f"已保存进度到: {OUTPUT_PARQUET}")

print("\n全部 batch 完成。")
print(f"最终输出: {OUTPUT_PARQUET}")

