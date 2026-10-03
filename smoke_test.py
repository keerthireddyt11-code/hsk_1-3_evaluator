from src.llm import assess, MODEL, BACKEND

TESTS = [
    "我的课程老师真酷。",              # expected correct (check if 课程/酷 are HSK 1-3!)
    "Wǒ de kèchéng lǎoshī zhēn kù.",   # pinyin input
    "我去商店昨天。",                  # wrong word order (time after place)
    "wo qu shangdian zuotian",         # toneless pinyin
    "asdf qwer zxcv",                  # gibberish (abstention test later)
]

print(f"Backend={BACKEND} Model={MODEL}\n")
for t in TESTS:
    result, raw, secs = assess(t)
    print(f"INPUT: {t}  ({secs:.1f}s)")
    print("  VALID JSON" if result else "  INVALID JSON")
    print(result.model_dump_json(indent=2) if result else raw)
    print("-" * 60)