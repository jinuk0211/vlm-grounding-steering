# Qwen baseline 선정과 pilot 결과 — 2026-10-26

선정한 baseline은 REVIS·RUDDER·GrAInS·VTI 네 개다. 공통 모델은 Qwen2.5-VL-7B-Instruct이며, 모델 revision·저자 commit·고정 프로필은 [선정 JSON](../configs/baseline_selection_20261026.json), 수치와 원본 검증파일 SHA는 [pilot 결과 JSON](results/qwen_pilot_20261026.json)에 기록했다.
이전 Qwen/Gemma 실행과 누적·실패 결과는 [전체 실험 결과 보고서](results/all_experiments_20261026_1954_KST.md)에 작성 시점 기준으로 보존했다.

상태와 점수는 보존된 검증 기록 기준이며 실시간 서버 관측이 아니다.

## REVIS·VTI가 '가장 높은 점수 버전'이라는 뜻인가?

소스 출처와 설정 선택을 구분해야 한다.

- REVIS 결과는 기존 원저자 함수 호출100쌍이다. 새 full Git clone과 실제 함수 AST·정규화된 소스·제공 Qwen 벡터가 일치한다. 논문 설정(alpha1.6/p80)과 공개 코드 설정(alpha2.5/p85)의 pilot 관측을 본 상태에서 공개 코드 프로필을 선정했다. 설정값 자체를 시험점수로 다시 튜닝하거나 가장 높은 중간 이정표만 뽑은 것은 아니다.
- VTI는 공개 코드의 literal VTILayer와 교정 mask를 사용한 기존 실행이다. 같은 fit50에서100강도 조합을 독립 개발20입력으로 평가해 caption과 질문 강도를 각각 선택했다. 개발 질문10개는 COCO 주석 기반이며 공식 POPE 시험문항이 아니다. 보고한62쌍은 시험 중 최고점이던 시점을 고른 결과가 아니라, 반복·미EOS 실패가 발생한 시점까지의 전체 결과다.
- 두 결과를 새 clone으로 새롭게100쌍 완주한 결과라고 부르지 않는다. Git clone 여부는 코드 확보 방식이고, 위 프로필 선정과 별개다. VTI의 남은38쌍 continuation은 완료되지 않았다.
- 이번 선정에는 pilot 결과가 참고됐으므로 추가 확인 평가에서는 설정을 고정하고 기존 pilot100개를 포함한 집계와 pilot을 제외한 집계를 구분한다. 현재 수치를 통계적으로 확정된 향상으로 해석하지 않는다.

## 방법당 하나씩 고른 프로필

| 방법 | 고정 버전 | 현재 사용 판정 |
|---|---|---|
| REVIS | 공개 코드: layer27 / alpha2.5 / percentile85, 저자 Qwen 벡터 | 완료100쌍 비교표에 우선 채택. 반복·512상한1건을 포함하며 full 적격은 별도 판단 |
| RUDDER | 새 clone Beta 기본: caption layer25/alpha5, POPE layer1/alpha8, k5/c1 | 완료100쌍 비교 baseline 채택. 개선이 없는 결과도 유지. 개발 보정 winner 아님 |
| GrAInS | 원본 tuple 계약: fit50 / IG20 / top50 / pos / PCA / layer27 / alpha10×공개scale10 / FP32 | 버전은 선정, 생성100쌍 결과 채택은 보류. nativeTensor 진단과 혼합하지 않음 |
| VTI | 교정 mask·개발선택: caption vision0.2/text0.5, POPE0.1/0.1; FP16·greedy·cache on | 버전은 선정,100쌍 완료 및 출력 실패 확인 전 최종 채택 보류 |

RUDDER/GrAInS의 import·연결 호환 변경과 GrAInS의 명시적 generation adapter 한계를 JSON에 남겼다. 저자 코드가 한 줄도 수정되지 않은 전체 실행으로 표시하지 않는다. 과거 RUDDER fallback과 GrAInS IG5/top20 CPU 진단은 별도 변형 결과이며 위 선정 프로필의 점수로 대체하지 않는다.

## 검증된 pilot 수치

단위는 %. 화살표는 각 프로필 자체의 같은 조건 Base→개입이다. 방법 간 서로 다른 Base를 한 대조군으로 합치지 않는다.

| 프로필 | 조건당 caption/POPE | 정확도 | 공식F1 | CHAIRs↓ | CHAIRi↓ | 객체재현율↑ |
|---|---:|---:|---:|---:|---:|---:|
| REVIS 공개 코드 |50/50|82→82|78.05→80.00|16→16|4.49→3.76|69.18→72.60|
| RUDDER 새 clone 기본 |50/50|82→80|78.05→76.19|24→26|5.13→5.90|70.55→70.55|
| GrAInS 원본 tuple |미시작|—|—|—|—|—|
| GrAInS nativeTensor 진단 |15/15|80→80|76.92→76.92|13.33→33.33|6.67→12.77|74.36→76.92|
| VTI 개발선택·실패 포함 |31/31|83.87→83.87|80→80|41.94→35.48|13.33→9.20|65.96→71.28|

REVIS의 문장 반복/512상한1건과 VTI의 문장 반복/512상한/미EOS1건을 점수에 남겼다. 두 사례 모두 자동 단어 반복 flag0과 원문 확인 반복1을 구분한다. RUDDER는 빈출력·진단반복·상한 모두0이다. GrAInS 진단은 Base 상한1건을 포함하며 원본 tuple 평가 결과가 아니다. CHAIR는 저장된 공식 문장별 객체 출력의 수치 재집계이며 새 classifier 실행으로 확대하지 않는다.

## 전체 평가 범위와 코드 통합 경계

전체 봉인 입력은 caption2000 + POPE9000(random/popular/adversarial 각3000) + MMHal96 = 11,096개다. fit50과 개발/보정 입력은 시험 개수에 포함하지 않는다. MMHal judge가 끝나지 않았으면 전체 채점 완료로 표시하지 않는다.

선정·설정·검증 점수 기록과 별도로 [RUDDER·GrAInS 원본 clone·실행 코드](../reproduction/clone_baselines_20261026/README.md)를 보존했다. 기존 CLI의 독립 adaptation 구현을 원저자 호출 실행기로 교체하지 않는다. 선정 JSON은 기존 CLI에 바로 전달하는 runtime 설정 파일이 아니다. 전체 네 방법의 재현 성공, 완료 full 성능, EPHS 우월성을 보고하지 않는다. 원문·가중치·데이터·대형 raw IG 백업은 GitHub에 포함하지 않는다.

## 고정 원저자 코드

- [REVIS 75f4d29](https://github.com/antgroup/Revis/tree/75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc)
- [RUDDER bd149d8](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-/tree/bd149d8dbde6a9097dd02aea80f83d0bd8876761)
- [GrAInS 8ad78db](https://github.com/duykhuongnguyen/GrAInS/tree/8ad78db3040fbb7b0aec9e544a3dcc7b9c050b5b)
- [VTI bb5dfee](https://github.com/shengliu66/VTI/tree/bb5dfee33a9e555f22af9faa7b3495c0ac3df69e)
