#!/usr/bin/env python3
"""설정 시점 1회용 SAM3 프롬프트 제안 도구.

bag(정지 장면)을 몇 프레임 뽑아 VLM(Qwen3-VL)에게 물체당 이름 후보 K개를
받고, SAM3로 그 이름 풀 전체를 검출해 같은 물체끼리 묶은 뒤 클러스터마다
SAM3 자신의 점수로 최고 이름을 고른다 — 근거는
`docs/vlm_prompt_spike_2026-09-04.md` "2차" 절(21/22 사례에서 VLM이 낸
이름만으로 수동 프롬프트와 동률 이상). 마지막 줄에 쉼표 구분 프롬프트
문자열만 찍는다 — 그대로 `run_offline.py --prompts` 나
`ros2 topic pub --once /perception/prompt` 에 넣는다.

VLM과 SAM3를 동시에 올리지 않는다(4B는 8.5~8.7GB로 SAM3와 12GB에 동거
불가 — 스파이크 §3) — VLM 후보를 다 받은 뒤 메모리를 비우고 SAM3를 올린다.

Usage:
  python3 scripts/suggest_prompts.py --bag bags/test4
  python3 scripts/suggest_prompts.py --bag bags/test2 --hand-prompts "thermos,laptop" --out output/suggest_test2.json
  python3 scripts/suggest_prompts.py --from-json output/suggest/test2.json  # bag/VLM/SAM3 없이 선택만 재실행
"""
import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_offline import read_bag  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "src" / "roboworld_perception"))
from roboworld_perception.prompt_suggest import (  # noqa: E402
    QUESTION_TMPL, best_per_name, choose_prompts, cluster_frame,
    find_collisions, link_across_frames, merge_overlapping_clusters,
    parse_candidates, summarize_cluster)

# 스파이크 §2-⑤: VLM은 대상이 아닌 것까지 다 열거한다(test4 2B는 클러스터
# 27개 중 실제 피킹 대상이 3개뿐이었다) — 질문에 이 지시문을 덧붙여 줄인다.
DEFAULT_INSTRUCTION = ("Only objects that a robot would pick; exclude the "
                       "conveyor, rollers, table, background and hands.")


def sample_frames(bag, every, max_frames, sync_slop):
    """read_bag 스트림에서 every 프레임마다 1장, 최대 max_frames장 (frames.py 스파이크와 동일 규약)."""
    frames = []
    for i, (_stamp_s, rgb, _depth, _K, _sync_ms) in enumerate(
            read_bag(bag, sync_slop=sync_slop)):
        if i % every != 0:
            continue
        frames.append(rgb)
        if len(frames) >= max_frames:
            break
    return frames


def resize_long_side(img, max_side=1024):
    """긴 변을 max_side로 축소한다 (확대는 하지 않음). vlm_common.resize_long_side 이식."""
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return img
    from PIL import Image
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)


def ask_vlm(model, processor, pil_image, prompt_text, max_new_tokens=512):
    """이미지 1장 + 텍스트 프롬프트로 생성 1회. vlm_common.ask 이식(동작 동일).

    Returns (text, latency_ms, peak_vram_mb_so_far).
    """
    import torch

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) * 1000

    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                    if torch.cuda.is_available() else 0.0)
    return text, latency_ms, peak_vram_mb


def build_pool(name_pool, hand_prompts):
    """VLM 이름 풀 + 수동 프롬프트를 합쳐 (정렬된 프롬프트 목록, name -> source). select.py의 build_pool 이식."""
    pool = {name: "vlm" for name in name_pool}
    for p in hand_prompts:
        key = p.strip().lower()
        if not key:
            continue
        pool[key] = "vlm+hand" if key in pool else "hand"
    return sorted(pool), pool


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default=None, help="--from-json 사용 시 불필요")
    ap.add_argument("--from-json", default=None,
                    help="저장된 --out JSON의 clusters/n_frames로 선택만 다시 돌린다 "
                         "(bag/VLM/SAM3 불필요) — 병합·선택 후 프롬프트 문자열만 찍는다")
    ap.add_argument("--every", type=int, default=30, help="이 프레임 수마다 1장 샘플 (기본 30)")
    ap.add_argument("--max-frames", type=int, default=12, help="최대 샘플 프레임 수 (기본 12)")
    ap.add_argument("--sync-slop", type=float, default=0.05,
                    help="color-depth 짝짓기 허용 시간차(초). Isaac bag은 올려야 한다")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct",
                    help="2B로 충분하다 — 4B는 SAM3와 12GB에 동거 불가(스파이크 §3)")
    ap.add_argument("--k", type=int, default=4, help="물체당 이름 후보 수 (기본 4)")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION,
                    help="VLM 질문 끝에 덧붙일 지시문 (빈 문자열이면 안 붙임)")
    ap.add_argument("--image-size", type=int, default=0, help="SAM3 입력 해상도 (0=기본 1008)")
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="SAM3 검출 임계값 (Sam3Detector 전용 — 선택 컷오프는 --min-score)")
    ap.add_argument("--min-score", type=float, default=0.75,
                    help="choose_prompts 선택 컷오프 (기본 0.75) — 실측 수동 프롬프트가 "
                         "0.87~0.97이라 그 아래는 잡음이었다")
    ap.add_argument("--out", default=None, help="클러스터·후보·score 전부를 남길 JSON 경로 (선택)")
    ap.add_argument("--hand-prompts", default="",
                    help="쉼표 구분 — 이름 풀에 비교용으로 합치되 source: hand로 표기")
    args = ap.parse_args()

    if args.from_json:
        data = json.loads(Path(args.from_json).read_text())
        clusters = merge_overlapping_clusters(data["clusters"])
        selected = choose_prompts(clusters, min_score=args.min_score,
                                  n_frames_total=data["n_frames"])
        if not selected:
            sys.exit("선택된 프롬프트가 없습니다 — --min-score를 낮추거나 "
                     f"{args.from_json}의 클러스터를 확인하세요.")
        print(",".join(s["selected_name"] for s in selected))
        return
    if not args.bag:
        ap.error("--bag이 필요합니다 (--from-json 사용 시 생략 가능)")

    print(f"[1/4] 프레임 추출 중... ({args.bag})", file=sys.stderr)
    frames = sample_frames(args.bag, args.every, args.max_frames, args.sync_slop)
    if not frames:
        sys.exit(f"프레임을 못 읽었습니다: {args.bag} (--sync-slop을 올려야 할 수 있습니다)")
    print(f"  {len(frames)}장 추출", file=sys.stderr)

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[2/4] VLM 로드 중... ({args.model})", file=sys.stderr)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    vlm_load_time_s = time.time() - t0

    question = QUESTION_TMPL.format(k=args.k)
    if args.instruction:
        question = f"{question} {args.instruction}"

    name_pool = Counter()
    vlm_latencies_ms = []
    vlm_peak_vram_mb = 0.0
    for i, rgb in enumerate(frames):
        img = resize_long_side(Image.fromarray(rgb))
        text, latency_ms, peak = ask_vlm(model, processor, img, question)
        objects = parse_candidates(text)
        frame_names = sorted({n.lower().strip() for o in objects for n in o["names"]})
        name_pool.update(frame_names)
        vlm_latencies_ms.append(latency_ms)
        vlm_peak_vram_mb = max(vlm_peak_vram_mb, peak)
        print(f"  프레임 {i + 1}/{len(frames)}: 물체 {len(objects)}개, 이름 {len(frame_names)}개",
              file=sys.stderr)

    print("  VLM 해제 중...", file=sys.stderr)
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hand_prompts = [p.strip() for p in args.hand_prompts.split(",") if p.strip()]
    prompts, pool_source = build_pool(name_pool, hand_prompts)
    if not prompts:
        sys.exit("이름 풀이 비어 있습니다 — VLM이 후보를 못 냈고 --hand-prompts도 없습니다.")

    print(f"[3/4] SAM3 로드 중... (이름 풀 {len(prompts)}개)", file=sys.stderr)
    from roboworld_perception.sam3_detector import Sam3Detector
    detector = Sam3Detector(threshold=args.threshold, image_size=args.image_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # VLM 해제 후 리셋 — SAM3만의 피크를 재려고

    frames_clusters = []
    sam_latencies_ms = []
    sam_peak_vram_mb = 0.0
    for i, rgb in enumerate(frames):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        dets = detector.detect(rgb, prompts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sam_latencies_ms.append((time.time() - t0) * 1000)
        if torch.cuda.is_available():
            sam_peak_vram_mb = max(sam_peak_vram_mb,
                                   torch.cuda.max_memory_allocated() / (1024 ** 2))
        frames_clusters.append(cluster_frame(best_per_name(dets)))
        print(f"  프레임 {i + 1}/{len(frames)}: 검출 {len(dets)}개", file=sys.stderr)

    print("[4/4] 클러스터링 및 선택 중...", file=sys.stderr)
    global_clusters = link_across_frames(frames_clusters)
    clusters = [summarize_cluster(i, gc, pool_source) for i, gc in enumerate(global_clusters)]
    clusters = merge_overlapping_clusters(clusters)
    clusters.sort(key=lambda c: -(c["selected_median_score"] or 0.0))
    collisions = find_collisions(clusters)
    selected = choose_prompts(clusters, min_score=args.min_score, n_frames_total=len(frames))
    prompt_str = ",".join(s["selected_name"] for s in selected)

    print(f"  클러스터 {len(clusters)}개, 충돌 {len(collisions)}개, 선택 {len(selected)}개",
          file=sys.stderr)

    if args.out:
        out_data = {
            "bag": args.bag, "model": args.model, "k": args.k,
            "instruction": args.instruction, "threshold": args.threshold,
            "min_score": args.min_score,
            "image_size": args.image_size, "hand_prompts": hand_prompts,
            "n_frames": len(frames), "name_pool": dict(name_pool.most_common()),
            "clusters": clusters, "collisions": collisions, "selected": selected,
            "prompts": prompt_str,
            "vlm_load_time_s": vlm_load_time_s,
            "vlm_latency_ms_median": (statistics.median(vlm_latencies_ms)
                                      if vlm_latencies_ms else None),
            "vlm_peak_vram_mb": vlm_peak_vram_mb,
            "sam_latency_ms_median": (statistics.median(sam_latencies_ms)
                                      if sam_latencies_ms else None),
            "sam_peak_vram_mb": sam_peak_vram_mb,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
        print(f"  -> {out_path}", file=sys.stderr)

    if not selected:
        sys.exit("선택된 프롬프트가 없습니다 — --min-score를 낮추거나 --out으로 "
                 "클러스터를 확인하세요 (드문 클러스터 필터·충돌 감점 때문일 수 있습니다).")
    print(prompt_str)


if __name__ == "__main__":
    main()
