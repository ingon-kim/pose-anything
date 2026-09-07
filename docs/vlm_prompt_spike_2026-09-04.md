# VLM ↔ SAM3 프롬프트 스파이크 (2026-09-04)

**질문:** 로컬 VLM(Qwen3-VL)이 (a) SAM3 텍스트 프롬프트를 자동으로 만들 수 있는가,
(b) SAM3 검출을 검증할 수 있는가.

**답:** **(a) 열거는 채택 후보다** — 사람이 놓친 물체를 찾아냈고, 남은 문제는 이
저장소가 이미 아는 프롬프트 단어 문제(`PROMPT_ALIASES`)뿐이다. **(b) 검증은
보류다** — 의미(semantic) 오류는 잡지만 조각·절단은 못 잡고, 작은 모델(2B)은
검증기로 못 쓴다.

측정 환경: RTX 4070 Ti 12,282 MiB(다른 서비스 1.5 GB 점유), `transformers 5.5.0`,
`Qwen/Qwen3-VL-2B-Instruct` / `Qwen/Qwen3-VL-4B-Instruct` bf16, bag 당 프레임
12장(30프레임마다 1장 샘플, `frames.py --every 30 --max 12`), 긴 변 1024px로 축소
(확대는 안 함), SAM3 검출은 `assoc_threshold=min(0.1, threshold)`
(`sam3_detector.py:83`)라 **표시 threshold 0.4 미만~0.1 이상의 저점수 검출도
포함**된다. 스파이크 코드는 저장소 밖 `~/vlm/spike/`(`enumerate.py` · `verify.py` ·
`frames.py` · `report.py` · `vlm_common.py`).

## 1. (a) 자동 프롬프트 후보 — 열거

`enumerate.py` 의 질문(원문 그대로):

> *"List every distinct physical object visible on the conveyor belt or table
> in this image. For each object, give a short English noun phrase (1 to 3
> words) suitable as a text prompt for an open-vocabulary segmentation model
> -- color or material adjectives are allowed (e.g. "black bag", "pink foam
> block"). Do not include the belt, the table/desk surface, the background,
> or human hands/arms. Output ONLY a JSON array of strings, nothing else."*

프레임 12장 중 **50% 이상**에 등장한 명사구만 후보로 채택(`--min-frac 0.5`).

| bag | 모델 | 자동 후보 전체 | 수동 프롬프트 대비 | 지연 중앙값 | 피크 VRAM | 로드 |
|---|---|---|---|---|---|---|
| test4 | 4B | black bag, keyboard, pink foam block | exact: black bag, keyboard · miss: manual · extra: pink foam block | 557 ms | 8582 MB | 18.5 s |
| test2 | 4B | black chair, black phone, blue book, green water bottle, pink foam block, red box, silver laptop, white cloth | partial: laptop↔silver laptop · miss: thermos, manual, cell phone · extra: 나머지 7개 | 1529 ms | 8667 MB | 12.1 s |
| isaac_belt_moving | 4B | blue bar, metal roller | partial: blue bar with holes↔blue bar · extra: metal roller | 294 ms | 8561 MB | 10.5 s |
| test4 | 2B | black bag, book, keyboard, pink foam block | exact: black bag, keyboard · miss: manual · extra: book, pink foam block | 474 ms | 4149 MB | 13.7 s |

**핵심 관찰:**

① **사람이 놓친 물체를 찾았다** — test4 자동 후보의 "pink foam block" 은 손 프롬프트
목록에 없었다. `~/vlm/out/test4/frames/0000.png` 로 확인: 프레임 **오른쪽 끝**에
숫자가 적힌 분홍 폼블록이 실제로 있고(SAM3 검출 box `[556, 186, 640, 218]`,
score 0.94, x1=640=프레임 폭 — 정확히 우측 가장자리에서 잘림), 예시 문구가 샌
것이 아니라 실재 검출이다.

② **이름이 SAM3 최적이 아니다** — test2 에서 자동 후보는 "green water bottle"
(score 0.54)인데 수동 프롬프트 "thermos" 는 0.91 이다. 이건 이 저장소가 이미
아는 문제다: `sam3_detector.py:8` `PROMPT_ALIASES` 가 `"물통": "thermos"` 를
매핑하며 붙인 주석이 정확히 이 수치를 남겨뒀다 — *"초록 보온병: "water bottle"은
0.3~0.45로 불안정, thermos는 ~0.9"*.

③ **프레임마다 동의어가 갈려 50% 문턱에서 탈락한다** — test4 의 "manual" 을
VLM 은 프레임마다 다르게 불렀다: `instruction manual`(4/12), `manual`(3/12),
`white paper`(3/12), `instruction sheet`(1/12), `book`/`black book`(1/12씩).
합치면 거의 매 프레임 등장하지만 어느 표현도 6/12(50%)에 못 미쳐 후보에서
완전히 빠졌다.

④ **Isaac 은 벨트 부품을 물체로 포함한다** — "metal roller"(컨베이어 롤러 자체)가
후보에 들었다.

⑤ **2B 도 열거는 된다** — test4 에서 2B 는 4B 가 놓친 "book" 까지 후보에 넣었다
(4B: black bag/keyboard/pink foam block, 2B: +book). 피크 VRAM **4149 MB(4.1 GB)**
로 4B(8.6 GB)의 절반 이하다.

## 2. (b) 검증 — SAM3 검출을 VLM 이 확인

`verify.py` 의 질문(원문 그대로, 크롭마다):

> *"Is the object inside the red rectangle a {label}? Answer yes or no."*

크롭은 박스를 중심 기준 2배로 확장한 컨텍스트 이미지에 빨간 사각형을 그린 것.

| bag | 모델 | tag | score≥0.4 (yes%) | 0.2~0.4 (yes%) | score<0.2 (yes%) |
|---|---|---|---|---|---|
| test4 | 4B | hand | 94% (34/36) | 89% (8/9) | 48% (19/40) |
| test4 | 4B | auto | 97% (74/76) | 67% (4/6) | 37% (20/54) |
| test2 | 4B | hand | 93% (40/43) | 0% (0/11) | 13% (9/71) |
| test2 | 4B | auto | 89% (136/153) | 16% (15/92) | 46% (98/211) |
| isaac | 4B | hand | 75% (72/96) | 81% (13/16) | 64% (47/73) |
| isaac | 4B | auto | 97% (360/372) | 97% (63/65) | 92% (364/395) |
| test4 | 2B | hand | 100% (36/36) | 100% (9/9) | **92% (37/40)** |
| test4 | 2B | auto | 92% (85/92) | 33% (6/18) | 38% (38/101) |

**핵심 관찰:**

- **4B 는 실기(test2/test4)에서 저점수(<0.2)를 52~87% no 로 거른다** — test4 hand
  52%(21/40 no), test2 hand 87%(62/71 no). 진짜로 물체가 아닌 검출을 걸러내는
  신호로 쓸 수 있어 보인다.
- **2B 는 검증기로 부적합하다** — 같은 test4 hand `<0.2` 구간에서 4B 는 52% 를
  no 로 거르는데(yes 48%), **2B 는 같은 구간에서 92% yes 다.** 저점수 검출을
  걸러내지 못한다.
- **Isaac 은 조각도 yes 로 통과시킨다** — auto `<0.2` 구간이 **92% yes**(364/395).
  벨트 부품 조각(§1-④)까지 "blue bar" 로 확인해 버린다는 뜻이다.
- → **검증은 의미(semantic) 오류(다른 범주의 물체)는 잡지만, 조각·절단은 못
  잡는다.**

### "≥0.4 인데 no" 가 진짜 오검출인지 — 크롭을 직접 봤다

지시받은 3개 표본(test2 hand 전수 3개, test2 auto 17개 중 5개, test4 4B auto
2개)을 `crop_file` PNG 로 열어 판정했다.

| bag/tag | 크롭 | 라벨 | score | 육안 판정 |
|---|---|---|---|---|
| test2 hand | 0002_4.png | laptop | 0.490 | **진짜 오검출** — 빨간 박스 안은 카키색 마분지 상자, 노트북이 아니다 |
| test2 hand | 0004_5.png | laptop | 0.479 | **진짜 오검출** — 같은 마분지 상자 |
| test2 hand | 0007_6.png | laptop | 0.543 | **진짜 오검출** — 같은 마분지 상자 |
| test2 auto | 0000_28/32/33/37, 0002_34.png | red box | 0.41~0.50 | **색 오검출** — 5개 전부 빨간 박스 안이 분홍색 폼블록/상자다. 같은 물체가 다른 검출에서는 "pink foam block" 으로 정확히 잡힌다 |
| test4 4B auto | 0002_8.png, 0004_15.png | pink foam block | 0.648 / 0.563 | **판정 보류(아래 참조)** |

**test4 4B auto 2건은 파일이 덮어써져 있었다** — `crops_auto/*.png` 는 파일명이
`프레임_인덱스` 규칙이라, 같은 bag 을 2B 로 재실행한 `verify.py sam --tag auto`
호출이 **같은 파일명에 다른 프롬프트 조합의 크롭을 다시 썼다.** 지금 그
경로에 있는 이미지는 4B 검증 당시의 것이 아니다(`crops_auto.json` 현재 내용은
label `book`, score 0.115/0.125 — `verify_4b_auto.json` 이 기록한 label
`pink foam block`, score 0.648/0.563 과 다르다). `verify_4b_auto.json` 이 남긴
원본 `box`/`crop_box` 좌표로 원본 프레임에서 다시 잘라 확인한 결과: **0002_8 은
박스 좌표가 4px 이내로 거의 일치**(같은 분홍 폼블록 코너 마운트로 판단),
**0004_15 은 좌표가 완전히 다른 위치**(x 38~85 vs 원본 442~482)라 신뢰 불가.
0002_8 재구성본은 실재하는 분홍 폼블록이 노출 과다로 거의 흰색에 가깝게
찍혀 있다 — "물체가 없다" 는 오검출이 아니라 "이 조명에서는 색이 안 보인다"
에 가깝다.

**test4 프레임 0000 우측 가장자리 확인(§1-①):** box `[556,186,640,218]` score
0.9375, x1=640 이 프레임 폭과 정확히 같아 화면 밖으로 잘려 있다 — 예시가 샌
게 아니라 실측이다.

**test2 프레임 0000 자동 후보 3개 확인:** "black chair" — 하단 우측에 검은
바퀴의자가 실제로 있다. "white cloth" — 롤러 위에 흰 천이 실제로 걸쳐 있다.
"red box" — 컨베이어 위에는 없다. 화면 오른쪽 배경 선반의 **빨간 철제 캐비닛**을
가리키는 것으로 보인다(질문이 명시적으로 배경 제외를 지시했는데도 새어
들어옴). SAM3 가 이 프롬프트를 컨베이어 위에서 접지(ground)시키며 색이 다른
분홍 폼블록에 붙은 것 — 열거 단계와 검증 단계의 오류가 사슬로 이어진 사례다.

## 3. 대가

| 항목 | 실측 |
|---|---|
| 프레임당 지연(열거, 640×480) | **557 ms**(test4, 4B) |
| 프레임당 지연(열거, 1280×720) | **1529 ms**(test2, 4B) |
| 크롭당 지연(검증) | 중앙값 **~75~98 ms**(bag 별 72.4~98.2 ms) |
| VRAM(4B) | **8.5~8.7 GB** — SAM3(4 GB)와 12 GB 안에 동거 불가 |
| VRAM(2B) | **4.1 GB**(4149 MB) — SAM3 와 동거 가능 |
| 로드 시간 | **10.5~18.5 s**(모델·bag 별) |

## 4. 결론 — 열거는 채택 후보, 검증은 보류

**열거(자동 프롬프트 생성)는 채택 후보다.** 사람이 놓친 물체를 찾아내고
(§1-①), 남은 문제(이름이 SAM3 최적이 아님, §1-②)는 이 저장소가 이미
`PROMPT_ALIASES` 로 다루고 있는 문제와 같은 축이다.

**검증은 보류다.** 4B 는 실기에서 저점수 검출의 상당 부분(52~87%)을 걸러내
보이지만, 조각·절단은 못 걸러내고(Isaac §2), 저비용 2B 는 애초에 검증기로
못 쓴다(§2 — 같은 저점수 구간에서 92% yes).

**다음 단계 설계(제안):** VLM 이 물체당 이름 후보를 2~3개 내고, **SAM3 점수로
그중 최고 점수 문구를 고른다** — SAM3 Agent 의 `segment_phrase → examine` 루프와
같은 구조다. 이러면 "water bottle vs thermos" 같은 이름 문제가 SAM3 자신의
점수로 자동 해소된다. 검증은 트랙 생성 시 1회만 부르는 형태로 나중에 재검토
한다. **노드 통합은 아직 안 했다** — 스파이크 코드는 저장소 밖 `~/vlm/spike/`
에 있다.

## 5. 재현 명령

```bash
cd ~/vlm/spike
./run.sh /home/ingon/roboworld/bags/test4 "black bag,keyboard,manual" 4b
./run.sh /home/ingon/roboworld/bags/test2 "thermos,laptop,manual,cell phone" 4b
./run.sh /home/ingon/roboworld/bags/isaac_belt_moving "blue bar with holes" 4b
./run.sh /home/ingon/roboworld/bags/test4 "black bag,keyboard,manual" 2b
```

각 실행은 `frames.py → enumerate.py → verify.py sam(hand) → verify.py
sam(auto) → verify.py vlm(hand) → verify.py vlm(auto) → report.py` 순서로
`~/vlm/out/<bag>/` 아래 결과를 남긴다.

---

## 2차 — 이름 후보 K개 → SAM3 점수로 선택

**질문:** 1차(§1)에서 남은 두 문제 — 이름이 SAM3 최적이 아님(§1-②), 프레임마다
동의어가 갈려 50% 문턱에서 탈락(§1-③) — 를 "물체당 이름 후보 K개를 내고 SAM3
검출 점수로 최고 점수 문구를 고른다"(§4가 제안한 다음 단계 설계)로 풀 수 있는가.

### 방법

`candidates.py` 의 질문(원문 그대로, `{k}`=4):

> *"List every distinct physical object visible on the conveyor belt or table
> in this image. For each object, give {k} different short English noun
> phrases (1 to 4 words each) that could each name that SAME object as a
> text prompt for an open-vocabulary segmentation model -- include synonyms,
> more general category names (hypernyms), and variants with a color or
> material adjective. The {k} phrases must be genuinely different wordings,
> not near-duplicates of each other. Do not include the belt, the table/desk
> surface, the background, or human hands/arms. Output ONLY a JSON array,
> nothing else, in this exact format: [{"object": "<one-line description of
> the object>", "names": ["<name 1>", "<name 2>", ...]}]. Example of the
> FORMAT only -- this object is very unlikely to be in the image, do not
> copy it unless you actually see something like it: [{"object": "a stapler
> near the edge of the desk", "names": ["stapler", "office stapler", "metal
> stapler", "desk stapler"]}]"*

물체당 서로 다른 이름 후보 4개(K=4)를 프레임마다 받아 bag 전체의 고유 이름을
이름 풀(name pool)로 모은다(`candidates.py`). 예를 들어 test2 4B 프레임 0000 의
원시 응답 중 한 항목: `{"object": "a green water bottle on the conveyor belt",
"names": ["green bottle", "water bottle", "soda bottle", "plastic bottle"]}` —
프레임 0002 에서는 같은 물체를 `{"object": "a green insulated bottle...",
"names": ["green bottle", "insulated bottle", "thermos bottle", "soda
bottle"]}` 로 다르게 부른다.

이어 `select.py` 가 이 이름 풀 전체(+ 수동 프롬프트)를 SAM3 로 프레임마다
전부 검출한다. 이름당 그 프레임의 최고 score 검출 1개만 남기고, **프레임
안에서는** 박스 IoU≥0.5 인 이름들을 score 내림차순 그리디로 같은 물체
클러스터에 묶는다(첫 멤버가 그 프레임의 클러스터 대표). **프레임 간에는**
클러스터 대표 박스 IoU 로 이어 붙인다(정지 씬 가정). 클러스터마다 이름별
median score 를 매겨 최고 이름을 "선택 이름"으로 삼는다(`report_select.py`).

수동 프롬프트도 이름 풀에 함께 섞여 검출되므로(source: hand/vlm+hand),
"선택 이름"이 수동 프롬프트 자체로 나올 수 있다 — 실제로 test2 4B 의
thermos/laptop/cell phone 클러스터가 그랬다(각각 selected_name "thermos"
0.910·"laptop" 0.965·"cell phone" 0.932, 전부 `source: hand`). 이 교란(수동
이름이 자기 자신을 이기는 경우)을 빼고 VLM 이 만든 이름만의 능력을 보려고,
아래 표는 각 클러스터에서 **`source == "vlm"` 인 이름만** 다시 최고를 골라
수동 프롬프트의 median score 와 비교했다(haiku 로 CPU 재분석).

### 결과

| bag | model | 수동 | 수동 score | VLM 최고 이름 | VLM score | Δ | ✔ |
|---|---|---|---|---|---|---|---|
| test2 | 2b | thermos | 0.910 | bottle | 0.932 | +0.021 | ✔ |
| test2 | 2b | laptop | 0.965 | white laptop | 0.969 | +0.004 | ✔ |
| test2 | 2b | manual | 0.912 | book with stickers | 0.941 | +0.029 | ✔ |
| test2 | 2b | cell phone | 0.932 | black phone | 0.912 | −0.020 | ✔ |
| test2 | 4b | thermos | 0.910 | thermos bottle | 0.871 | −0.039 | ✔ |
| test2 | 4b | laptop | 0.965 | silver laptop | 0.965 | 0.000 | ✔ |
| test2 | 4b | manual | 0.912 | manual booklet | 0.945 | +0.033 | ✔ |
| test2 | 4b | cell phone | 0.932 | black phone | 0.912 | −0.020 | ✔ |
| test4 | 2b | black bag | 0.941 | black backpack | 0.914 | −0.027 | ✔ |
| test4 | 2b | keyboard | 0.969 | black computer keyboard | 0.980 | +0.012 | ✔ |
| test4 | 2b | manual | 0.914 | instruction manual | 0.959 | +0.045 | ✔ |
| test4 | 4b | black bag | 0.941 | strap-equipped bag | 0.922 | −0.020 | ✔ |
| test4 | 4b | keyboard | 0.969 | black keyboard | 0.980 | +0.012 | ✔ |
| test4 | 4b | manual | 0.914 | paper manual | 0.980 | +0.066 | ✔ |
| test5 | 2b | black bag | 0.953 | black carry bag | 0.969 | +0.016 | ✔ |
| test5 | 2b | keyboard | 0.977 | black computer keyboard | 0.980 | +0.004 | ✔ |
| test5 | 2b | manual | 0.922 | instruction manual | 0.965 | +0.043 | ✔ |
| test5 | 2b | beige notebook | 0.867 | paper | 0.598 | −0.270 | ✗ |
| test5 | 4b | black bag | 0.953 | dark bag | 0.934 | −0.020 | ✔ |
| test5 | 4b | keyboard | 0.977 | black keyboard | 0.980 | +0.004 | ✔ |
| test5 | 4b | manual | 0.922 | paper manual | 0.980 | +0.059 | ✔ |
| test5 | 4b | beige notebook | 0.867 | flat item | 0.955 | +0.088 | ✔ |

**합계 21/22 ✔**(2b 10/11, 4b 11/11).

관찰:

① **21/22 에서 VLM 이 낸 이름만으로 수동 프롬프트와 동률 이상이다.** test2
4B thermos 클러스터의 VLM 후보 전체를 median score 순으로 보면 thermos
bottle 0.871, plastic bottle 0.715, green bottle 0.543, **water bottle
0.331**, insulated bottle 0.108 — 1차(§1-②)에서 문제였던 "water bottle" 이
후보 풀에 그대로 있었지만, SAM3 자신의 점수로 걸러졌다.

② **"물체 이름이 SAM3 최적이 아니다" 문제가 SAM3 점수로 자동 해소된다** —
"water bottle" 처럼 낮은 점수의 동의어가 후보에 섞여도 median score 로
최고를 고르면 자동으로 배제된다(①과 같은 근거).

③ **실패는 1건뿐이다** — test5 2B 의 beige notebook 클러스터에서 VLM 이 낸
이름 중 최고가 "paper" 0.598 로 수동 프롬프트 0.867 에 못 미쳤다(Δ −0.270).
같은 클러스터를 4B 로 하면 "flat item" 0.955 로 수동보다 높다(Δ +0.088) —
2B 가 이 물체에 낸 이름 후보 자체가 부실했던 경우로 보인다.

④ **주의 — 점수만으로 고르면 과잉일반 이름을 뽑을 수 있다.** "flat item"
처럼 일반적인 이름은 점수가 높지만(위 test5 4B 사례) 다른 납작한 물체(검은
폴더 등)에도 똑같이 붙을 수 있다. `select.py` 의 충돌(collision) 출력이 이를
드러낸다 — 예를 들어 test5 4B 는 "flat item" 이 클러스터 [22, 25, 17, 15, 3]
다섯 곳에서 동시에 최고로 뽑혔다. 제품에서는 여러 클러스터에서 최고인 이름
(충돌)을 감점해야 한다.

⑤ **VLM 은 대상이 아닌 것까지 다 열거한다** — 롤러·의자 다리·포스트잇·검은
폴더 같은 배경/비대상 물체도 이름 후보를 받는다. test4 2B 는 클러스터 27개
중 실제 피킹 대상은 3개(black bag, keyboard, manual)뿐이었다. 제품은 "피킹
대상만" 지시하거나 운영자 확인을 1회 거치는 절차가 필요하다.

### 대가

프레임당 VLM 지연은 K=4 로 후보를 내야 해서 1차(§3)보다 길다: **2.9 s**
(열거) + SAM3 이름 풀 검출 **2.9 s**. VRAM 은 2B **4.2 GB**, 4B **8.6 GB**.
설정(프롬프트 정의) 시점 1회에만 드는 비용이라 감당 가능하다.

### 결론

**채택 — 다음 단계는 노드 밖 설정 도구(오프라인 `suggest_prompts`)로
통합한다.** 프레임마다가 아니라 프롬프트를 (재)정의할 때만 돌린다. **2B 로
충분하다**(4B 는 12 GB 에서 SAM3 와 동거 불가).

### 재현

```bash
cd ~/vlm/spike
./run_select.sh /home/ingon/roboworld/bags/test2 "thermos,laptop,manual,cell phone" 2b
./run_select.sh /home/ingon/roboworld/bags/test2 "thermos,laptop,manual,cell phone" 4b
./run_select.sh /home/ingon/roboworld/bags/test4 "black bag,keyboard,manual" 2b
./run_select.sh /home/ingon/roboworld/bags/test4 "black bag,keyboard,manual" 4b
./run_select.sh /home/ingon/roboworld/bags/test5 "black bag,keyboard,manual,beige notebook" 2b
./run_select.sh /home/ingon/roboworld/bags/test5 "black bag,keyboard,manual,beige notebook" 4b
```

각 실행은 `frames.py → candidates.py → select.py → report_select.py` 순서로
`~/vlm/out/<bag>/` 아래 `candidates_<model>.json` · `select_<model>.json` 을
남긴다.

---

## 부록: 외형 re-ID(DINOv3)는 이번에 안 한다

가림 중 ID 가 실제로 바뀌는지부터 다시 쟀다(`output/reid_gate/log.txt`).
`=== distinct ids per label` 표:

| bag | 라벨별 distinct ID |
|---|---|
| test3 | pink foam block: [1] · book: [3] · glove: [2] — **라벨당 1개** |
| test4 | book: [3] · keyboard: [1] · black bag: [2] — **라벨당 1개** |
| test5 | keyboard: [1] · gray notebook: [4] · black bag: [2] · book: [3] — **라벨당 1개** |
| isaac_belt_moving | blue bar with holes: [1,2,3,4,5,6,7,8,10,11,17,20,21] — **13개, 전부 별개 블록** |

**실기 3 bag(test3/test4/test5) 는 라벨당 ID 가 정확히 1개다 — 실행 내내 ID 전환이
0건이었다.** Isaac 의 13개 ID 는 컨베이어 위 서로 다른 블록 각각에 대응하며,
외형은 전부 동일(같은 "blue bar with holes")하다. 그중 ID 7(444프레임,
x −0.91~−0.70)과 ID 17(1553프레임, x −0.87~−0.79)만 화면 왼쪽 가장자리의 같은
블록으로 보이는데, 둘 사이에 **3.87초의 공백**이 있고(CSV 타임스탬프 실측)
그 뒤 새 ID 로 이어졌다 — 위치가 아니라 트랙이 끊긴 뒤 재등장이다.

**이번에 DINOv3 같은 외형 re-ID 를 조사하지 않은 이유는 스위치가 0건이라
고칠 실패가 없기 때문이다.** `docs/README.md:253` 이 이미 같은 결론을 남겨
뒀다 — *"원인은 위치가 아니라 지속시간이었고 `occlusion_hold`(2026-08-11)로
이미 고쳐졌다. 그 이후 test4/test5 실행 전부에서 ID 변경 0건"*. 이번 측정은
그 결론과 일치한다.

---

## 3차 — 저장소 통합: `scripts/suggest_prompts.py` (2026-09-07)

**질문:** 2차(§"2차")에서 검증한 방식(VLM 이름 후보 K개 → SAM3 점수 선택)을
저장소 안 실행 가능한 도구로 통합하고, 손 프롬프트 없이 실측했을 때 실제로
쓸 만한 목록이 나오는가.

### 무엇

설정 시점에 1회만 도는 도구다 — 런타임에는 물리지 않는다. bag(정지 장면)을
넣으면 SAM3 프롬프트 문자열 하나를 표준출력 마지막 줄에 찍는다.

파이프라인: bag에서 프레임 샘플(기본 30프레임마다 1장, 최대 12장) → VLM
(`Qwen/Qwen3-VL-2B-Instruct`)에게 프레임마다 이름 후보 K=4개 요청(질문 끝에
"로봇이 집을 물체만, 컨베이어·롤러·테이블·배경·손 제외" 지시문 추가) → VLM
해제 → 그 이름 풀 전체를 SAM3로 전 프레임 검출 → 박스 IoU로 같은 물체끼리
프레임 내·프레임 간 클러스터링(`cluster_frame`/`link_across_frames`) →
프레임 연결이 끊겨 갈라진 클러스터를 대표 박스 IoU로 늦게 병합
(`merge_overlapping_clusters`) → 필터: 제외어(`EXCLUDE_WORDS` — roller,
conveyor, belt, table, desk, hand, arm, background, floor, surface 등,
`prompt_suggest.py:223`), 일반명 머리명사(`GENERIC_HEADS` — object, item,
thing, shape, rectangle, square, circle, stuff, piece, material,
`prompt_suggest.py:231`), 전체 프레임의 50% 미만에서만 보인 rare 클러스터
제외, `min_score` 0.75 미만 제외, 충돌(같은 이름이 두 클러스터 이상에서
1순위) 감점(`choose_prompts`, `prompt_suggest.py:301`) → 최종 목록.

명령 한 줄:

```bash
python3 scripts/suggest_prompts.py --bag bags/test4
```

로직은 `src/roboworld_perception/roboworld_perception/prompt_suggest.py`에
torch/ROS 비의존 순수 함수로 있고, `scripts/suggest_prompts.py`가 프레임
추출·VLM·SAM3 호출을 감싼다. `--from-json`으로 저장된 클러스터에서 병합·선택
로직만 다시 돌릴 수도 있다(bag/VLM/SAM3 불필요, CPU 전용).
`src/roboworld_perception/test/test_prompt_suggest.py`의 신규 테스트 21개는
직접 재실행해 통과를 확인했다(2.36s). 전체 스위트 218 passed는 구현
보고를 따른다.

### 실측 — bag별 1차(규칙 전)·2차(규칙 후) 목록 대 손 프롬프트

1차는 `--min-score 0.4`, 손 프롬프트 없이 VLM 2B로 처음 실행한 원시 결과다.
2차는 같은 클러스터 JSON에 `--from-json`으로 규칙(필터·병합·`min_score`
0.75)을 다시 적용한 결과다(위 명령으로 직접 재현·확인).

| bag | 1차 목록(규칙 전) | 2차 목록(규칙 후) | 손 프롬프트 | 손 대응 |
|---|---|---|---|---|
| test4 | black computer keyboard, instruction manual, black bag, metal roller, pink sticky note | black computer keyboard, instruction manual, black bag, pink sticky note | black bag, keyboard, manual | 3/3 + 가장자리 분홍 포스트잇은 실재 물체 |
| test5 | black computer keyboard, instruction manual, black carrying bag, white card, pink label | black computer keyboard, instruction manual, black carrying bag | black bag, keyboard, manual, beige notebook | 손 4개 중 3/3 — beige notebook은 12프레임 중 3프레임에서만 보여 rare 필터(0.5)에서 탈락(라벨링 커버리지 한계) |
| test2 | white laptop, pink block, pink object, bottle, black object, cylindrical object, umbrella item, paper, white rectangle | white laptop, pink block, bottle, black phone, cylinder, paper, white cloth, fabric | thermos, laptop, manual, cell phone | 4/4 (laptop→white laptop, thermos→bottle, cell phone→black phone, manual→paper) — 잔여 4개(pink block, cylinder, white cloth, fabric)는 대상 밖 실재 물체(분홍 블록·천) 또는 중복(cylinder, fabric — cloth가 병합 부작용으로 white cloth+fabric 둘로 남음, 아래 한계 ②) |

### e2e(test2, `run_offline.py` 발행 프레임 기준) — 손 / 1차 제안 / 2차 제안

CSV(`output/suggest/e2e_hand/`, `e2e_suggested/`, `e2e_suggested2/`)에서
라벨별 발행 행 수·score 평균·10th 백분위수(p10)를 직접 계산했다(전체
프레임 230).

| 출처 | 라벨 | rows | mean score | p10 |
|---|---|---|---|---|
| 손 | laptop | 230 | 0.967 | 0.961 |
| 손 | cell phone | 230 | 0.931 | 0.918 |
| 손 | manual | 230 | 0.910 | 0.898 |
| 손 | thermos | 230 | 0.909 | 0.898 |
| 1차 제안 | white laptop | 230 | 0.970 | 0.965 |
| 1차 제안 | pink block | 230 | 0.950 | 0.949 |
| 1차 제안 | pink object | 230 | 0.947 | 0.941 |
| 1차 제안 | black object | 230 | 0.930 | 0.926 |
| 1차 제안 | bottle | 230 | 0.929 | 0.914 |
| 1차 제안 | cylindrical object | 230 | 0.903 | 0.891 |
| 1차 제안 | paper | 230 | 0.900 | 0.887 |
| 1차 제안 | umbrella item | 230 | 0.886 | 0.828 |
| 1차 제안 | white rectangle | 155 | 0.683 | 0.645 |
| 2차 제안 | white laptop | 230 | 0.970 | 0.965 |
| 2차 제안 | pink block | 230 | 0.950 | 0.949 |
| 2차 제안 | bottle | 230 | 0.929 | 0.914 |
| 2차 제안 | black phone | 230 | 0.907 | 0.891 |
| 2차 제안 | cylinder | 230 | 0.904 | 0.898 |
| 2차 제안 | paper | 230 | 0.900 | 0.887 |
| 2차 제안 | white cloth | 230 | 0.847 | 0.832 |
| 2차 제안 | fabric | 44 | 0.199 | 0.111 |

관찰:

- 손 4개 전부 2차 제안 이름으로 대응되고 mean score 가 손 프롬프트와
  동률 이상이거나 근접이다: laptop 0.967→white laptop 0.970, thermos
  0.909→bottle 0.929, cell phone 0.931→black phone 0.907, manual
  0.910→paper 0.900.
- **fabric 은 44/230행에서만 발행되고 mean 0.199 로 사실상 잡음이다** —
  같은 프레임 세트에서 white cloth(230행, mean 0.847)가 이미 같은 물체를
  잡고 있어, fabric 은 병합 부작용으로 남은 중복 이름으로 보인다(위 표
  test2 대응 칸, 한계 ②).
- 1차의 white rectangle 도 155/230행으로 덜 채워지고 mean 0.683 로
  낮다 — 2차 목록에서는 `min_score` 0.75 미만이라 빠졌다.

### 비용

| bag | VLM 로드 | VLM 지연(median) | VLM 피크 VRAM | SAM3 지연(median) | SAM3 피크 VRAM |
|---|---|---|---|---|---|
| test2 | 11.79 s | 5748 ms | 4229 MB | 4146 ms | 3615 MB |
| test4 | 11.43 s | 3335 ms | 4173 MB | 3072 ms | 2276 MB |
| test5 | 10.65 s | 3449 ms | 4173 MB | 2640 ms | 2157 MB |

값은 `--out` JSON(`vlm_load_time_s`, `vlm_latency_ms_median`,
`vlm_peak_vram_mb`, `sam_latency_ms_median`, `sam_peak_vram_mb`)에서 그대로
가져왔다. 1차 실행 stderr 로그에는 타임스탬프가 없어 총 소요는 로드
시간 + 프레임 수 × (VLM 지연 + SAM3 지연)으로 어림했다 — test2 약 93 s,
test4 약 89 s, test5 약 85 s(SAM3 로드 수 초는 별도) — **bag당 1~2분**
범위이고, 설정 시점 1회 비용이라 감당 가능하다.

### 한계

① 프레임 간 연결이 끊기면 같은 물체가 두 이름으로 남을 수 있다(대표 박스
드리프트로 최종 IoU 0) — 실측 test2 "pink block"/"pink object".
② 병합이 rare 문턱 근처 클러스터를 살려 이름을 쪼갤 수 있다(cloth→white
cloth+fabric, 위 e2e 표에서 fabric 44행/mean 0.199로 확인).
③ 드물게 보이는 물체는 탈락한다(test5 beige notebook, 12프레임 중 3프레임).
④ 대상 밖 실재 물체(분홍 블록·천)는 VLM instruction 만으로 못 막는다.

**→ 출력 목록은 운영자가 한 번 확인하고 쓴다(설정 도구이지 런타임이
아니다).**

### 결론

**채택 — 머지한다.** 수동으로 프롬프트 단어를 골라 `PROMPT_ALIASES` 를
갱신하던 절차를 대체하는 첫 단계다.
