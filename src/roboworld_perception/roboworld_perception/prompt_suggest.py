"""VLM 이름 후보 -> SAM3 점수 선택 — 순수 로직 (torch/ROS import 없음).

`docs/vlm_prompt_spike_2026-09-04.md` "2차" 절에서 검증한 방식을 그대로
옮긴다: VLM(Qwen3-VL)이 물체당 이름 후보 K개를 내면, SAM3로 이름 풀 전체를
검출해 같은 물체끼리 박스 IoU로 묶고(`cluster_frame`/`link_across_frames`),
클러스터마다 이름별 median score로 최고 이름을 고른다(`summarize_cluster`).
21/22 사례에서 VLM이 낸 이름만으로 수동 프롬프트와 동률 이상이었다(2b 10/11,
4b 11/11) — "물통"이 "water bottle"(0.3~0.45)보다 "thermos"(~0.9)로 훨씬 잘
검출되는 것과 같은 축의 문제가, 후보를 여러 개 내고 SAM3 자신의 점수로
고르면 자동으로 해소된다. 다만 **주의점 둘**: ① 점수만으로 고르면 "flat
item"처럼 여러 클러스터에 동시에 최고로 뽑히는 과잉일반 이름이 나올 수
있다(`find_collisions`가 드러낸다 — 예: test5 4B에서 클러스터 5곳이 동시에
"flat item"을 골랐다) — `choose_prompts`의 충돌 감점은 이를 다음 순위 이름으로
대체해 완화한다. ② VLM은 대상이 아닌 것까지 다 열거한다(test4 2B는 클러스터
27개 중 실제 피킹 대상이 3개뿐이었다) — 그래서 호출 측(`scripts/suggest_prompts.py`)
의 질문에는 "로봇이 집을 물체만" 지시하는 문장을 덧붙이고, `choose_prompts`는
드물게만 보이는 클러스터(배경 잡음)를 프레임 비율로 걸러낸다.

2차(2026-09-07, 손 프롬프트 없이 첫 실측): instruction을 안 지킨 이름
("metal roller")과 과잉일반 이름("black object", "cylindrical object")이
그대로 통과하는 문제가 드러나 `EXCLUDE_WORDS`/`GENERIC_HEADS` 필터를
추가했고, 프레임 연결이 끊겨 갈라진 클러스터를 늦게 구제하는
`merge_overlapping_clusters`를 추가했다(단, 최종 rep_box가 드리프트로
안 겹치면 구제 못 함 — `merge_overlapping_clusters` 참고). `min_score`
기본값도 0.4->0.75로 올렸다.
"""
import json
import re
import statistics


# ── candidates.py 이식 ────────────────────────────────────────────────

QUESTION_TMPL = (
    "List every distinct physical object visible on the conveyor belt or table "
    "in this image. For each object, give {k} different short English noun "
    "phrases (1 to 4 words each) that could each name that SAME object as a "
    "text prompt for an open-vocabulary segmentation model -- include "
    "synonyms, more general category names (hypernyms), and variants with a "
    "color or material adjective. The {k} phrases must be genuinely different "
    "wordings, not near-duplicates of each other. Do not include the belt, "
    "the table/desk surface, the background, or human hands/arms. "
    "Output ONLY a JSON array, nothing else, in this exact format: "
    "[{{\"object\": \"<one-line description of the object>\", "
    "\"names\": [\"<name 1>\", \"<name 2>\", ...]}}]. "
    "Example of the FORMAT only -- this object is very unlikely to be in the "
    "image, do not copy it unless you actually see something like it: "
    "[{{\"object\": \"a stapler near the edge of the desk\", "
    "\"names\": [\"stapler\", \"office stapler\", \"metal stapler\", "
    "\"desk stapler\"]}}]"
)


def parse_candidates(text: str):
    """모델 응답에서 [{"object":..,"names":[...]}, ...] 파싱. 정규식 폴백.

    2단 폴백(직접 JSON -> 실패 시 "[...]" 정규식 추출 후 재파싱)으로 모델이
    답 앞뒤에 잡담을 붙여도 건진다. 형식이 안 맞는 원소는 조용히 버린다 —
    이 함수는 절대 예외를 내지 않고, 실패하면 빈 리스트를 돌려준다.
    """
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = None
    if not isinstance(parsed, list):
        return []

    objects = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        names = item.get("names")
        if not isinstance(names, list):
            continue
        names = [str(n).strip() for n in names if str(n).strip()]
        if not names:
            continue
        objects.append({"object": str(item.get("object", "")).strip(), "names": names})
    return objects


# ── select.py 이식 ────────────────────────────────────────────────────

CLUSTER_IOU = 0.5


def box_iou(a, b):
    """IoU of xyxy boxes.

    roboworld_perception.tracker.box_iou와 같은 6줄짜리 순수 함수다. 그
    모듈은 .geometry를 통해 open3d/scipy를 끌고 오므로, 이 모듈은 torch/ROS
    없이 단독으로 쓸 수 있게 로컬 복사로 둔다.
    """
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def best_per_name(dets):
    """한 프레임의 검출 리스트 -> {name: {"box":.., "score":..}} (이름당 최고 score만)."""
    best = {}
    for d in dets:
        name = d["label"]
        score = float(d["score"])
        if name not in best or score > best[name]["score"]:
            best[name] = {"box": [float(v) for v in d["box"]], "score": score}
    return best


def cluster_frame(name_boxes):
    """프레임 안에서 이름들을 박스 IoU>=CLUSTER_IOU로 그리디 클러스터링.

    score 내림차순으로 처리해 각 클러스터의 대표(첫 멤버)가 그 프레임에서
    가장 신뢰도 높은 검출이 되게 한다. 기존 클러스터 중 IoU가 가장 크고
    문턱을 넘는 곳에 붙이고, 없으면 새 클러스터를 연다.
    """
    items = sorted(name_boxes.items(), key=lambda kv: -kv[1]["score"])
    clusters = []  # [{"rep_box":.., "members": [(name, box, score), ...]}]
    for name, d in items:
        best_idx, best_iou = None, 0.0
        for i, c in enumerate(clusters):
            iou = box_iou(c["rep_box"], d["box"])
            if iou >= CLUSTER_IOU and iou > best_iou:
                best_idx, best_iou = i, iou
        entry = (name, d["box"], d["score"])
        if best_idx is None:
            clusters.append({"rep_box": d["box"], "members": [entry]})
        else:
            clusters[best_idx]["members"].append(entry)
    return clusters


def link_across_frames(frames_clusters):
    """프레임별 클러스터 리스트를 대표 박스 IoU로 이어붙인다 (프레임마다 그리디 전역 할당).

    정지 씬 가정 — 같은 물체는 프레임이 바뀌어도 위치가 거의 고정이다.
    문턱을 못 넘으면 새 전역 클러스터로 열린다(별 클러스터) — frame_ids
    개수가 전체 프레임 수보다 적으면 어딘가 끊긴 것.
    """
    global_clusters = []  # [{"rep_box":.., "per_name": {name: [score, ...]}, "frame_ids": set()}]
    for frame_idx, fclusters in enumerate(frames_clusters):
        pairs = []
        for fi, fc in enumerate(fclusters):
            for gi, gc in enumerate(global_clusters):
                iou = box_iou(fc["rep_box"], gc["rep_box"])
                if iou >= CLUSTER_IOU:
                    pairs.append((iou, fi, gi))
        pairs.sort(key=lambda p: -p[0])
        assigned_f, assigned_g, assign = set(), set(), {}
        for iou, fi, gi in pairs:
            if fi in assigned_f or gi in assigned_g:
                continue
            assign[fi] = gi
            assigned_f.add(fi)
            assigned_g.add(gi)

        for fi, fc in enumerate(fclusters):
            if fi in assign:
                gc = global_clusters[assign[fi]]
                gc["rep_box"] = fc["rep_box"]  # 최신 프레임 대표 박스로 갱신
            else:
                gc = {"rep_box": fc["rep_box"], "per_name": {}, "frame_ids": set()}
                global_clusters.append(gc)
            gc["frame_ids"].add(frame_idx)
            for name, _box, score in fc["members"]:
                gc["per_name"].setdefault(name, []).append(score)
    return global_clusters


def summarize_cluster(cluster_id, gc, pool_source):
    names = []
    for name, scores in gc["per_name"].items():
        names.append({
            "name": name,
            "source": pool_source.get(name, "vlm"),
            "median_score": statistics.median(scores),
            "n_frames": len(scores),
        })
    names.sort(key=lambda n: -n["median_score"])
    return {
        "cluster_id": cluster_id,
        "box": [round(v, 1) for v in gc["rep_box"]],
        "n_frames": len(gc["frame_ids"]),
        "names": names,
        "selected_name": names[0]["name"] if names else None,
        "selected_median_score": names[0]["median_score"] if names else None,
    }


def find_collisions(clusters):
    """같은 selected_name이 두 개 이상의 클러스터에서 최고로 뽑힌 경우."""
    by_name = {}
    for c in clusters:
        if c["selected_name"] is None:
            continue
        by_name.setdefault(c["selected_name"], []).append(c["cluster_id"])
    return [{"name": n, "cluster_ids": ids} for n, ids in by_name.items() if len(ids) > 1]


# ── 실측 후 선택 규칙 강화 (2차) ──────────────────────────────────────
#
# `docs/` 실측(2026-09-07, test2/test4/test5 — 손 프롬프트 없이 VLM 2B 이름
# 풀 그대로 SAM3에 태움)에서 드러난 두 문제:
#   ① VLM이 instruction을 안 지키고 벨트 부품 등을 후보로 낸다
#      (test4 "metal roller" 0.89 — instruction에 conveyor/roller 제외라고
#      명시했는데도 나옴).
#   ② 점수만으로 이름을 고르면 "black object"/"cylindrical object"처럼
#      일반명(과잉일반)이 실제 물체명(같은 클러스터의 "black phone"
#      0.912, "cylinder" 0.904)보다 앞서 뽑힌다 — find_collisions의 충돌
#      감점은 "같은 이름이 두 클러스터의 1순위"일 때만 발동해 이 경우는
#      못 잡는다.
# EXCLUDE_WORDS/GENERIC_HEADS가 그 둘을 막는다.

EXCLUDE_WORDS = frozenset({
    "roller", "conveyor", "belt", "table", "desk", "hand", "arm",
    "background", "floor", "surface",
})

# 마지막 단어(머리 명사)가 이 목록이면 "과잉일반" 이름으로 보고 다음 후보로
# 넘어간다. "block"은 넣지 않는다 — test2 실측에서 "pink block"은 실제
# 물체명이고, 같은 클러스터의 "pink object"/"pink item"만 걸러야 한다.
GENERIC_HEADS = frozenset({
    "object", "item", "thing", "shape", "rectangle", "square", "circle",
    "stuff", "piece", "material",
})


def _words(name):
    return re.findall(r"[a-z0-9]+", name.lower())


def has_excluded_word(name):
    """EXCLUDE_WORDS 중 하나라도 단어 단위로 포함하면 True."""
    return any(w in EXCLUDE_WORDS for w in _words(name))


def is_generic_name(name):
    """머리 명사(마지막 단어)가 GENERIC_HEADS에 있으면 True."""
    words = _words(name)
    return bool(words) and words[-1] in GENERIC_HEADS


def merge_overlapping_clusters(clusters):
    """대표 박스 IoU >= CLUSTER_IOU인 클러스터 쌍을 병합한다.

    `link_across_frames`의 프레임 간 연결이 끊겨 같은 물체가 전역 클러스터
    둘로 갈라지는 경우를 늦게라도 구제한다. 이름 풀을 합치고(같은 이름이
    양쪽에 있으면 median_score가 더 높은 쪽만 남긴다) n_frames는 합산한다
    (서로 다른 프레임 부분집합을 커버했다는 가정). `choose_prompts`보다
    먼저, 그리고 (rare-cluster 필터 전에) 호출해야 한다 — 이미 드문
    클러스터로 걸러진 뒤에는 병합해도 못 구제한다.

    한 쌍만 그리디로 병합한다(3개 이상 연쇄 겹침은 다루지 않음 — 실측
    3개 bag에 그런 사례가 없었다).

    주의: 같은 물체가 실제로 갈라진 경우에도 rep_box가 프레임마다 최신
    값으로 갱신되며 드리프트하면, 그때그때는 IoU가 문턱을 넘었어도 최종
    rep_box끼리는 안 겹칠 수 있다 — 이 함수는 그런 경우를 못 잡는다(실측
    test2 "pink block"/"pink object", 최종 IoU 0 — 대신 GENERIC_HEADS가
    "pink object"/"pink item" 클러스터를 통째로 걸러 해소한다).
    """
    remaining = list(clusters)
    used = [False] * len(remaining)
    merged = []
    for i, base in enumerate(remaining):
        if used[i]:
            continue
        used[i] = True
        combined = {n["name"]: dict(n) for n in (base.get("names") or [])}
        n_frames = base["n_frames"]
        for j in range(i + 1, len(remaining)):
            if used[j] or box_iou(base["box"], remaining[j]["box"]) < CLUSTER_IOU:
                continue
            used[j] = True
            other = remaining[j]
            n_frames += other["n_frames"]
            for n in (other.get("names") or []):
                if (n["name"] not in combined
                        or n["median_score"] > combined[n["name"]]["median_score"]):
                    combined[n["name"]] = dict(n)
        names = sorted(combined.values(), key=lambda d: -d["median_score"])
        merged.append({
            **base, "n_frames": n_frames, "names": names,
            "selected_name": names[0]["name"] if names else None,
            "selected_median_score": names[0]["median_score"] if names else None,
        })
    return merged


# ── 최종 선택 (신규) ──────────────────────────────────────────────────

def choose_prompts(clusters, min_score=0.75, min_frames_frac=0.5, *, n_frames_total=None):
    """클러스터마다 최종 프롬프트 이름 하나를 골라 score 내림차순으로 반환.

    각 항목은 {"cluster_id", "selected_name", "score", "n_frames"}. 규칙:
    (a) 최종 이름의 median score < min_score 인 클러스터는 뺀다. min_score
        기본값 0.75 — 실측(수동 프롬프트 0.87~0.97)상 그 아래는 잡음이었다
        (예: "white card" 0.67, "pink label" 0.61).
    (b) 전체 프레임의 min_frames_frac 미만에서만 보인 클러스터는 뺀다
        (배경 잡음 — 스파이크 §2-⑤: VLM은 대상이 아닌 것까지 다 열거한다).
        같은 단계에서 EXCLUDE_WORDS를 포함하거나 GENERIC_HEADS인 이름은
        후보 목록에서 미리 걸러낸다 — 다음 순위 이름으로 자동 대체되고,
        남는 이름이 없으면 클러스터째 제외된다(실측 test4 "metal roller",
        test2 "black object"->"black phone", "pink object"/"pink item"만
        남은 클러스터 제외).
    (c) 충돌 감점 — 같은 이름이 둘 이상의 클러스터에서 1순위(최고 median
        score)면 그 이름은 과잉일반이다(스파이크 §2-④ "flat item" 사례).
        해당 클러스터들 전부 그 이름을 빼고 각자 2순위 이름으로 대체한다
        (한 번만 — 1순위 기준으로 딱 한 번 검사·대체한다). 2순위가 없으면
        (a)에서 자연히 빠진다.
    (d) 최종 이름 중복 제거 — (c)는 원래(1순위) 이름만 보고 딱 한 번
        대체하므로, 서로 다른 충돌 그룹에서 대체된 이름이 우연히 같아지는
        경우(연쇄 충돌)를 못 잡는다. 그래서 마지막에 한 번 더 — 같은 최종
        이름이 남으면 score가 더 높은 클러스터만 남긴다.

    (c) 검사 전에 1순위 자체가 이미 min_score 미만인 클러스터는 미리 뺀다 —
    names가 median_score 내림차순이라 1순위가 최댓값이므로, 그게 이미
    min_score 미만이면 어차피 (a)에서 빠질 클러스터다. 미리 빼지 않으면
    그런 죽은 클러스터의 1순위 이름이 우연히 진짜 높은 점수 클러스터와
    겹칠 때 정상 클러스터를 엉뚱하게 감점시킨다.

    n_frames_total: 전체 프레임 수(select 단계가 처리한 프레임 개수).
    키워드 전용 인자다 — min_score를 위치 인자로 넘겨도 실수로 여기 섞여
    들어가지 않는다. 생략(None)하면 클러스터 중 n_frames 최댓값으로
    추정한다 — 정지 씬에서는 보통 배경(벨트/롤러 등)이 거의 모든
    프레임에서 잡히므로 근사가 맞지만, 호출 측이 실제 처리 프레임 수를
    알고 있다면(예: `select` 단계의 `n_frames`) 그 값을 넘기는 쪽이 정확하다.
    """
    if n_frames_total is None:
        n_frames_total = max((c["n_frames"] for c in clusters), default=0)

    # (b) 배경 잡음 제외 + 제외어/일반어 이름 걸러내기
    kept = []
    for c in clusters:
        if n_frames_total and c["n_frames"] / n_frames_total < min_frames_frac:
            continue
        names = [n for n in (c.get("names") or [])
                 if not has_excluded_word(n["name"]) and not is_generic_name(n["name"])]
        if not names:
            continue
        kept.append({"cluster_id": c["cluster_id"], "n_frames": c["n_frames"],
                     "names": names, "rank": 0})

    # names가 median_score 내림차순이라 names[0]이 그 클러스터의 최댓값 —
    # 이게 이미 min_score 미만이면 어떤 순위로도 못 살아나 (a)에서 반드시
    # 빠진다. (c) 충돌 검사 전에 미리 빼야, 어차피 죽을 클러스터의 1순위
    # 이름이 진짜 높은 점수 클러스터를 엉뚱하게 감점시키는 걸 막는다.
    kept = [cand for cand in kept if cand["names"][0]["median_score"] >= min_score]

    # (c) 충돌 감점 — 1순위(rank 0) 이름 기준으로 딱 한 번 검사해 대체
    by_top_name = {}
    for cand in kept:
        by_top_name.setdefault(cand["names"][0]["name"], []).append(cand)
    for cand in kept:
        cand["rank"] = 1 if len(by_top_name[cand["names"][0]["name"]]) > 1 else 0

    # (a) 저점수/소진 클러스터 제외
    selected = []
    for cand in kept:
        if cand["rank"] >= len(cand["names"]):
            continue
        entry = cand["names"][cand["rank"]]
        if entry["median_score"] < min_score:
            continue
        selected.append({"cluster_id": cand["cluster_id"], "selected_name": entry["name"],
                         "score": entry["median_score"], "n_frames": cand["n_frames"]})

    # (d) 최종 이름 중복 제거 — score 내림차순으로 먼저 만난 것만 남긴다
    selected.sort(key=lambda s: -s["score"])
    seen_names = set()
    result = []
    for s in selected:
        if s["selected_name"] in seen_names:
            continue
        seen_names.add(s["selected_name"])
        result.append(s)
    return result
