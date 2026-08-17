import os
import sys
import re

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

sys.path.insert(0, r"c:\Users\Dell\Desktop\fyp\backend")

from llm_service import OllamaLLM

llm = OllamaLLM(model_name="qwen2.5:3b")

contexts = [
    {
        "content": (
            "Early blight in tomato is caused by Alternaria solani. "
            "Symptoms include dark brown spots with concentric rings on older leaves forming a target-board pattern. "
            "Leaves turn yellow around spots and eventually die. "
            "Favorable conditions include warm temperatures between 24 and 29 degrees Celsius and high humidity. "
            "Chemical control: Apply mancozeb or chlorothalonil fungicide every 7 to 10 days. "
            "Organic control: Spray copper-based fungicide or neem oil solution. "
            "Prevention: Use disease-free seeds, rotate crops, avoid overhead irrigation. "
            "Treatment: Remove infected leaves immediately and apply appropriate fungicide."
        ),
        "metadata": {"source_file": "tomato_diseases.pdf"}
    }
]

answer = llm.generate_answer(
    query="Give complete information about Early blight in tomato.",
    contexts=contexts
)

print("=" * 60)
print("FORMATTED OUTPUT (what the app receives):")
print("=" * 60)
print(answer)
print()

print("=" * 60)
print("VALIDATION RESULTS:")
print("=" * 60)

truly_forbidden = ["#", "*", "_", "/", ","]
found_bad = []
for ch in truly_forbidden:
    if ch in answer:
        found_bad.append(repr(ch))

line_start_dash = re.search(r"^- ", answer, re.MULTILINE)
if line_start_dash:
    found_bad.append("'- ' (line-start dash marker)")

if found_bad:
    print(f"FAIL  - Forbidden characters/patterns still present: {found_bad}")
else:
    print("PASS  - No forbidden markdown characters or line-start dashes found")
    print("        (hyphens inside compound words like disease-free are allowed)")

bullet_char = "\u2022"
if bullet_char in answer:
    print("PASS  - Bullet character (bullet) is present in output")
else:
    print("INFO  - Bullet not found; model may use plain newlines instead")

heading_match = re.search(r"^#{1,6}", answer, re.MULTILINE)
if heading_match:
    print("FAIL  - Markdown headings (#) found in output")
else:
    print("PASS  - No markdown headings found")

print()
print("Raw repr (first 500 chars):")
print(repr(answer[:500]))
