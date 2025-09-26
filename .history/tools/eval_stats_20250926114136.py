# tools/eval_stats.py
import json, numpy as np

def pctl(arr, q):
    if not arr: return None
    return float(np.percentile(np.array(arr), q))

def run(log_file="logs/app3.jsonl"):
    voice_total, voice_asr, voice_llm, voice_tts = [], [], [], []
    skill_hits = {}

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except:
                continue
            if rec.get("event") == "voice_turn":
                voice_total.append(rec.get("total_ms", 0))
                voice_asr.append(rec.get("asr_ms", 0))
                voice_llm.append(rec.get("llm_ms", 0))
                voice_tts.append(rec.get("tts_ms", 0))
                sk = rec.get("skill") or "none"
                skill_hits[sk] = skill_hits.get(sk, 0) + 1
            elif rec.get("event") == "chat_turn":
                sk = rec.get("skill") or "none"
                skill_hits[sk] = skill_hits.get(sk, 0) + 1

    print("== 延迟统计（毫秒）==")
    if voice_total:
        print(f"Voice total P50={pctl(voice_total,50):.0f}, P95={pctl(voice_total,95):.0f}")
        print(f"  ASR   P50={pctl(voice_asr,50):.0f}, P95={pctl(voice_asr,95):.0f}")
        print(f"  LLM   P50={pctl(voice_llm,50):.0f}, P95={pctl(voice_llm,95):.0f}")
        print(f"  TTS   P50={pctl(voice_tts,50):.0f}, P95={pctl(voice_tts,95):.0f}")
    else:
        print("暂无 voice_turn 记录")

    print("\n== 技能触发计数 ==")
    for k,v in sorted(skill_hits.items(), key=lambda kv: -kv[1]):
        print(f"{k:15s}: {v}")

if __name__ == "__main__":
    run()
