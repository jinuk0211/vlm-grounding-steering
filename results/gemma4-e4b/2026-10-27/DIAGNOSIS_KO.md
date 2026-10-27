# POPE 결과가 거의 그대로인 이유: 확인된 사실과 남은 진단

2026-10-27. 사용자 요청으로 CHAIR 생성과 자동 실행을 일시 정지했다. 완료된 POPE 결과와 진행 중인 CHAIR 응답은 보존했다. GPU0의 일시 정지 프로세스는 재개를 위해 메모리를 유지한다.

## 실행한 코드의 정체

clone한 `jinuk0211/vlm-grounding-steering`은 공통 실험 저장소다. 실행된 네 baseline은 각 논문 저자 저장소를 그대로 호출하는 구현이 아니다. 저장소 HEAD `1683e86223cdc491a2223d61ec63403bf37f7513`의 `src/vlm_grounding/baselines.py` 첫 문장부터 `Paper-derived, local baseline implementations (not author reproductions)`라고 명시돼 있다. 이번 산출물에도 `paper-derived-local-adaptation`, `author_reproduction: false`를 기록했다.

예를 들어 REVIS의 벡터 추출·보정·prefill 위치 범위와 VTI의 PCA·주입 연산이 저자 공개 구현과 다르다. 로컬 `docs/baseline_audit_2026-10-23_ko.md`와 `docs/baseline_implementation.md`에 비교가 있다. VTI 저자 코드는 정규화한 입력과 방향을 섞고 기존 norm을 복원하지만 로컬 VTI는 방향을 더한다. 이 차이가 이번 점수 정체의 원인이라고 확정한 것은 아니다.

- [REVIS 저자 hook, 고정 커밋](https://github.com/antgroup/Revis/blob/75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc/steering_utils/llm_layers.py)
- [VTI 저자 hook, 고정 커밋](https://github.com/shengliu66/VTI/blob/bb5dfee33a9e555f22af9faa7b3495c0ac3df69e/vti_utils/llm_layers.py)

실행 코드에는 추가로 GrAInS IG의 BF16 비유한 gradient 문제를 막기 위한 수치 적분 변경이 있다. 전체 코드를 변경 없이 실행했다고 표현할 수 없다. 해당 변경과 코드 해시는 프로토콜에 남아 있다.

## 실제 3,000문항에서 바뀐 답

완료된 `merged/*.jsonl`을 문항 ID로 정렬해 Base와 각 방법의 Yes/No 응답을 독립 비교했다.

| 방법 | Base와 답이 다른 문항 | 오답→정답 | 정답→오답 | 정답 순증감 | 정확도 |
|---|---:|---:|---:|---:|---:|
| Base | 0 | 0 | 0 | 0 | 83.8333% |
| REVIS | 21 | 12 | 9 | +3 | 83.9333% |
| RUDDER | 4 | 1 | 3 | −2 | 83.7667% |
| GrAInS | 134 | 52 | 82 | −30 | 82.8333% |
| VTI | 33 | 15 | 18 | −3 | 83.7333% |

점수가 거의 같은 직접적인 이유는 세 방법에서 최종 Yes/No가 드물게 바뀌고, 그중 개선과 악화가 상쇄됐기 때문이다. GrAInS는 더 많은 답을 바꿨지만 악화가 더 많았다. 이 결과만으로 hook의 모든 실행 경로나 개입 크기가 올바르다고 증명할 수는 없다.

## 현재 설정과 해석 범위

Gemma E4B, COCO128의 별도 개발 이미지 32장으로 추출하고 REVIS는 추가 32장으로 보정했다. 기존 벡터를 벤치마크 간 재사용했다. REVIS alpha=1.6, RUDDER layer=23/alpha=0.5, GrAInS alpha=0.5/IG 16단계, VTI text/vision alpha=0.5이며 Gemma용 최적화가 끝난 설정이 아니다. POPE는 greedy Yes/No 생성이다.

확인된 결론은 이 Gemma 적용본과 고정 설정에서 전반적인 개선을 확인하지 못했다는 것이다. 저자 원본의 발표 성능 재현 결과로 취급할 수 없다. 구현 차이, 개입 크기, 보정 상태의 영향 중 무엇이 주원인인지는 아직 분리 검증하지 않았다.

다음 진단에서는 대량 평가 재개에 앞서 별도 개발 입력으로 저자 연산과 로컬 연산의 hook 위치·gate·개입 전후 norm·첫 답변 logits를 대조해야 한다. 테스트 정답으로 강도를 골라 기존 점수를 끌어올리는 방식은 사용하지 않는다.

POPE 원본 결과: `/home/project-b/LJW/vlm-grounding-steering/outputs/pope-selected-20261027-161933`.
