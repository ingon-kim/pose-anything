"""prompt_suggest.py 단위 테스트 (CPU, torch/모델 없음).

docs/vlm_prompt_spike_2026-09-04.md "2차" 절에서 검증한 로직을 그대로
옮긴 모듈이라, 여기서는 그 근거가 된 구체적 수치·시나리오를 합성 데이터로
재현한다."""
import pytest

from roboworld_perception.prompt_suggest import (choose_prompts, cluster_frame,
                                                  link_across_frames,
                                                  merge_overlapping_clusters,
                                                  parse_candidates)


# ── parse_candidates ──────────────────────────────────────────────────

def test_parse_candidates_normal_json():
    text = ('[{"object": "a green water bottle", '
            '"names": ["green bottle", "water bottle", "soda bottle"]}]')
    out = parse_candidates(text)
    assert out == [{"object": "a green water bottle",
                    "names": ["green bottle", "water bottle", "soda bottle"]}]


def test_parse_candidates_with_surrounding_chatter():
    """모델이 답 앞뒤에 잡담을 붙여도 정규식 폴백으로 건진다."""
    text = ('Sure, here is the JSON you asked for:\n'
            '[{"object": "a stapler", "names": ["stapler", "office stapler"]}]\n'
            'Let me know if you need anything else!')
    out = parse_candidates(text)
    assert out == [{"object": "a stapler", "names": ["stapler", "office stapler"]}]


def test_parse_candidates_malformed_returns_empty():
    assert parse_candidates("this is not json at all {[") == []
    assert parse_candidates('{"object": "not a list"}') == []
    assert parse_candidates('[{"object": "no names key"}]') == []
    assert parse_candidates('[{"object": "empty names", "names": []}]') == []


# ── cluster_frame / link_across_frames ────────────────────────────────

def test_cluster_frame_same_object_two_names_one_cluster():
    """같은 물체를 가리키는 두 이름은 박스가 겹치면 한 클러스터로 묶인다."""
    name_boxes = {
        "green bottle": {"box": [100, 100, 140, 200], "score": 0.6},
        "thermos": {"box": [102, 101, 142, 199], "score": 0.9},
        "keyboard": {"box": [300, 300, 460, 340], "score": 0.95},
    }
    clusters = cluster_frame(name_boxes)
    assert len(clusters) == 2
    bottle_cluster = next(c for c in clusters
                          if {"thermos", "green bottle"} <= {m[0] for m in c["members"]})
    assert {m[0] for m in bottle_cluster["members"]} == {"thermos", "green bottle"}
    # score 내림차순으로 처리하므로 대표(rep_box)는 thermos(0.9) 쪽 박스
    assert bottle_cluster["rep_box"] == [102, 101, 142, 199]
    keyboard_cluster = next(c for c in clusters if c is not bottle_cluster)
    assert {m[0] for m in keyboard_cluster["members"]} == {"keyboard"}


def test_link_across_frames_connects_same_object_across_frames():
    """정지 씬 — 프레임이 바뀌어도 같은 물체는 위치가 거의 고정, 전 프레임에 연결된다."""
    frame1 = cluster_frame({
        "thermos": {"box": [100, 100, 140, 200], "score": 0.9},
        "keyboard": {"box": [300, 300, 460, 340], "score": 0.95},
    })
    frame2 = cluster_frame({
        "thermos": {"box": [101, 100, 141, 200], "score": 0.87},
        "bottle": {"box": [100, 99, 140, 199], "score": 0.93},
        "keyboard": {"box": [301, 300, 461, 341], "score": 0.96},
    })
    global_clusters = link_across_frames([frame1, frame2])
    assert len(global_clusters) == 2  # 두 물체 -> 전역 클러스터 2개, 안 끊김

    bottle_gc = next(gc for gc in global_clusters if "thermos" in gc["per_name"])
    assert bottle_gc["frame_ids"] == {0, 1}
    assert bottle_gc["per_name"]["thermos"] == [0.9, 0.87]
    assert bottle_gc["per_name"]["bottle"] == [0.93]  # frame2에만 등장

    keyboard_gc = next(gc for gc in global_clusters if "keyboard" in gc["per_name"])
    assert keyboard_gc["frame_ids"] == {0, 1}
    assert keyboard_gc["per_name"]["keyboard"] == [0.95, 0.96]


def test_link_across_frames_unmatched_object_starts_new_cluster():
    """겹치는 박스가 없으면 새 전역 클러스터로 열린다(별 클러스터)."""
    frame1 = cluster_frame({"book": {"box": [0, 0, 50, 50], "score": 0.8}})
    frame2 = cluster_frame({"book": {"box": [500, 500, 550, 550], "score": 0.8}})
    global_clusters = link_across_frames([frame1, frame2])
    assert len(global_clusters) == 2
    assert {frozenset(gc["frame_ids"]) for gc in global_clusters} == {frozenset({0}),
                                                                       frozenset({1})}


# ── choose_prompts ─────────────────────────────────────────────────────

def _cluster(cluster_id, n_frames, names):
    """names: [(name, median_score), ...] -> summarize_cluster 형태 클러스터."""
    name_dicts = [{"name": n, "source": "vlm", "median_score": s, "n_frames": n_frames}
                  for n, s in names]
    name_dicts.sort(key=lambda d: -d["median_score"])
    return {"cluster_id": cluster_id, "box": [0, 0, 1, 1], "n_frames": n_frames,
            "names": name_dicts,
            "selected_name": name_dicts[0]["name"] if name_dicts else None,
            "selected_median_score": name_dicts[0]["median_score"] if name_dicts else None}


def test_choose_prompts_excludes_low_score():
    """(a) score < min_score 클러스터 제외."""
    clusters = [
        _cluster(0, n_frames=12, names=[("keyboard", 0.98)]),
        _cluster(1, n_frames=12, names=[("blurry thing", 0.2)]),
    ]
    out = choose_prompts(clusters, n_frames_total=12, min_score=0.4)
    names = {c["selected_name"] for c in out}
    assert names == {"keyboard"}


def test_choose_prompts_excludes_rare_cluster():
    """(b) 전체 프레임의 min_frames_frac 미만에서만 보인 클러스터 제외(배경 잡음)."""
    clusters = [
        _cluster(0, n_frames=12, names=[("black bag", 0.94)]),
        _cluster(1, n_frames=2, names=[("sticky note", 0.9)]),  # 12장 중 2장뿐
    ]
    out = choose_prompts(clusters, n_frames_total=12, min_score=0.4, min_frames_frac=0.5)
    names = {c["selected_name"] for c in out}
    assert names == {"black bag"}


def test_choose_prompts_collision_demotes_both_to_second_rank():
    """(c) 같은 이름이 두 클러스터에서 최고 -> 둘 다 2순위 이름으로 대체.

    "flat sheet"(머리 명사 "sheet", GENERIC_HEADS 아님)로 이름을 잡아
    일반어 필터와 충돌 감점을 분리해서 본다."""
    clusters = [
        _cluster(0, n_frames=12, names=[("flat sheet", 0.95), ("beige notebook", 0.6)]),
        _cluster(1, n_frames=12, names=[("flat sheet", 0.9), ("black folder", 0.7)]),
    ]
    out = choose_prompts(clusters, n_frames_total=12, min_score=0.4)
    by_cluster = {c["cluster_id"]: c["selected_name"] for c in out}
    assert by_cluster == {0: "beige notebook", 1: "black folder"}
    assert all(c["selected_name"] != "flat sheet" for c in out)


def test_choose_prompts_low_score_cluster_does_not_poison_collision():
    """저점수(어차피 min_score에 못 미쳐 탈락할) 클러스터의 1순위 이름이
    진짜 고득점 클러스터와 우연히 겹쳐도, 그 고득점 클러스터를 감점시키면
    안 된다 — names[0]이 이미 min_score 미만이면 그 클러스터는 사전에
    제외된다."""
    clusters = [
        _cluster(0, n_frames=12, names=[("flat sheet", 0.95), ("beige notebook", 0.6)]),
        _cluster(1, n_frames=12, names=[("flat sheet", 0.15)]),  # 배경 잡음, 어차피 탈락
    ]
    out = choose_prompts(clusters, n_frames_total=12, min_score=0.4)
    assert len(out) == 1
    assert out[0]["selected_name"] == "flat sheet"  # 감점 없이 1순위 그대로 유지
    assert out[0]["score"] == 0.95


def test_choose_prompts_keyword_only_n_frames_total():
    """min_score를 위치 인자로 넘겨도 n_frames_total과 뒤섞이지 않는다
    (n_frames_total은 키워드 전용)."""
    clusters = [_cluster(0, n_frames=12, names=[("keyboard", 0.98)])]
    out = choose_prompts(clusters, 0.4, n_frames_total=12)
    assert [c["selected_name"] for c in out] == ["keyboard"]
    with pytest.raises(TypeError):
        choose_prompts(clusters, 0.4, 0.5, 12)  # n_frames_total은 위치 인자로 못 받음


def test_choose_prompts_reproduces_spike_test2_scenario():
    """스파이크 test2 시나리오: bottle 0.93 vs thermos 0.91 vs green water
    bottle 0.54 -> 최고 점수 "bottle"이 선택된다(§2-①, water bottle류
    저점수 동의어는 SAM3 자신의 점수로 자동 배제됨을 재현). min_score는
    생략해 기본값(0.75, 2차 실측 이후 상향)으로 통과하는지 함께 본다."""
    clusters = [_cluster(0, n_frames=12, names=[("bottle", 0.93), ("thermos", 0.91),
                                                ("green water bottle", 0.54)])]
    out = choose_prompts(clusters, n_frames_total=12)
    assert len(out) == 1
    assert out[0]["selected_name"] == "bottle"
    assert out[0]["score"] == 0.93


# ── 2차(2026-09-07 실측 후) — EXCLUDE_WORDS / GENERIC_HEADS / merge / 기본 min_score ──

def test_choose_prompts_default_min_score_now_0_75():
    """(5) min_score 기본값이 0.4->0.75로 올라갔다 — 실측(수동 프롬프트
    0.87~0.97)상 "white card" 0.67 같은 저점수는 잡음이었다."""
    clusters = [
        _cluster(0, n_frames=12, names=[("keyboard", 0.98)]),
        _cluster(1, n_frames=12, names=[("white card", 0.67)]),
    ]
    out = choose_prompts(clusters, n_frames_total=12)  # min_score 생략 -> 기본값
    assert [c["selected_name"] for c in out] == ["keyboard"]


def test_choose_prompts_generic_head_falls_through_to_next_candidate():
    """(2) 머리 명사가 일반어(GENERIC_HEADS)면 다음 비일반어 후보로 대체
    — 실측 test2 "black object"/"black item" -> "black phone". "pink
    block"(머리 명사 "block")은 일반어가 아니라 그대로 유지된다."""
    clusters = [
        _cluster(0, n_frames=12, names=[("black object", 0.93), ("black item", 0.926),
                                        ("black phone", 0.912)]),
        _cluster(1, n_frames=12, names=[("pink block", 0.949)]),
    ]
    out = choose_prompts(clusters, n_frames_total=12, min_score=0.75)
    by_cluster = {c["cluster_id"]: c["selected_name"] for c in out}
    assert by_cluster == {0: "black phone", 1: "pink block"}


def test_choose_prompts_generic_head_exhausted_drops_cluster():
    """일반어 이름만 남으면(대체할 비일반어 후보가 없으면) 클러스터째
    제외된다 — 실측 test2 "pink object"/"pink item" 뿐인 클러스터(프레임
    연결이 끊겨 "pink block"과 별개로 갈라진 것)."""
    clusters = [
        _cluster(0, n_frames=8, names=[("pink block", 0.949)]),
        _cluster(1, n_frames=4, names=[("pink object", 0.949), ("pink item", 0.941)]),
    ]
    out = choose_prompts(clusters, n_frames_total=8, min_score=0.75)
    assert [c["selected_name"] for c in out] == ["pink block"]


def test_choose_prompts_excludes_word_from_pool():
    """(3) EXCLUDE_WORDS를 포함하는 이름은 풀에서 제외 — 실측 test4 "metal
    roller"(instruction에서 벨트/롤러 제외하라 했는데도 VLM이 후보로 냄).
    이 클러스터는 대체할 이름이 없어 통째로 제외된다."""
    clusters = [_cluster(0, n_frames=9, names=[("metal roller", 0.895),
                                               ("metal roller on belt", 0.187)])]
    out = choose_prompts(clusters, n_frames_total=12, min_score=0.75)
    assert out == []


def test_merge_overlapping_clusters_combines_iou_pair():
    """(4) 대표 박스 IoU>=0.5인 클러스터 쌍은 병합된다 — 실측 test5
    cluster 2(7프레임)/cluster 12(3프레임), 둘 다 "instruction manual",
    실제 IoU~0.56. n_frames는 합산되고 median_score는 더 높은 쪽이 남는다."""
    clusters = [
        {"cluster_id": 2, "box": [418.0, 172.0, 488.0, 262.0], "n_frames": 7,
         "names": [{"name": "instruction manual", "source": "vlm",
                    "median_score": 0.961, "n_frames": 7}]},
        {"cluster_id": 12, "box": [388.0, 149.0, 488.0, 262.0], "n_frames": 3,
         "names": [{"name": "instruction manual", "source": "vlm",
                    "median_score": 0.965, "n_frames": 3}]},
    ]
    merged = merge_overlapping_clusters(clusters)
    assert len(merged) == 1
    assert merged[0]["n_frames"] == 10
    assert merged[0]["names"][0]["median_score"] == 0.965


def test_merge_overlapping_clusters_no_overlap_stays_separate():
    """겹치지 않으면 병합하지 않는다 — 실측 test2 "pink block"/"pink
    object" 클러스터의 최종 대표 박스는 IoU 0(프레임마다 갱신되며 드리프트
    했기 때문) — merge로는 못 구제되고, 대신 일반어 필터가 처리한다
    (test_choose_prompts_generic_head_exhausted_drops_cluster 참고)."""
    clusters = [
        {"cluster_id": 1, "box": [362.0, 231.0, 516.0, 282.0], "n_frames": 8, "names": []},
        {"cluster_id": 2, "box": [13.4, 592.0, 154.0, 660.0], "n_frames": 4, "names": []},
    ]
    merged = merge_overlapping_clusters(clusters)
    assert len(merged) == 2


def test_full_pipeline_reproduces_test2_pink_block_split():
    """실측 test2 재현: "pink block"(클러스터 1, 8프레임)과 "pink
    object"/"pink item" 뿐인 클러스터(클러스터 2, 4프레임)로 프레임 연결이
    끊겨 갈라짐. 두 대표 박스는 안 겹쳐(IoU 0) merge_overlapping_clusters로는
    못 구제하지만, 클러스터 2는 이름이 전부 일반어라 choose_prompts가
    통째로 제외해 결과적으로 중복이 사라진다."""
    clusters = merge_overlapping_clusters([
        {"cluster_id": 1, "box": [362.0, 231.0, 516.0, 282.0], "n_frames": 8,
         "names": [{"name": "pink block", "source": "vlm",
                    "median_score": 0.949, "n_frames": 8}]},
        {"cluster_id": 2, "box": [13.4, 592.0, 154.0, 660.0], "n_frames": 4,
         "names": [{"name": "pink object", "source": "vlm",
                    "median_score": 0.949, "n_frames": 4},
                   {"name": "pink item", "source": "vlm",
                    "median_score": 0.941, "n_frames": 1}]},
    ])
    out = choose_prompts(clusters, n_frames_total=8, min_score=0.75)
    assert [c["selected_name"] for c in out] == ["pink block"]


def test_choose_prompts_dedupes_final_names():
    """(d) 최종 이름 중복 제거 — (c)는 1순위 이름만 보고 한 번만 대체하므로,
    서로 다른 충돌 그룹에서 대체된 이름이 우연히 같아지는 연쇄 충돌은 못
    잡는다. A/C가 "gadget"으로 충돌해 A는 2순위 "widget"으로 밀리는데,
    B는 원래부터 1순위가 "widget"이었다 -> A와 B가 "widget"에서 다시
    부딪힌다. (d)가 이를 잡아 score 높은 쪽(B, 0.55)만 남긴다."""
    clusters = [
        _cluster("A", n_frames=12, names=[("gadget", 0.9), ("widget", 0.5)]),
        _cluster("B", n_frames=12, names=[("widget", 0.55)]),
        _cluster("C", n_frames=12, names=[("gadget", 0.8), ("thingamajig", 0.6)]),
    ]
    out = choose_prompts(clusters, n_frames_total=12, min_score=0.4)
    by_cluster = {c["cluster_id"]: c["selected_name"] for c in out}
    assert by_cluster == {"B": "widget", "C": "thingamajig"}  # A는 중복 제거로 탈락


def test_choose_prompts_sorted_by_score_descending():
    clusters = [
        _cluster(0, n_frames=12, names=[("manual", 0.8)]),
        _cluster(1, n_frames=12, names=[("keyboard", 0.98)]),
    ]
    out = choose_prompts(clusters, n_frames_total=12, min_score=0.4)
    assert [c["selected_name"] for c in out] == ["keyboard", "manual"]
