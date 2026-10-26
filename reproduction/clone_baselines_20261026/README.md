# RUDDER·GrAInS 원본 clone과 Qwen 실행 기록

이 디렉터리는 2026-10-26 선정 기록에 대응하는 **코드 출처**다. 직전 PR 버전은 선정 문서·설정·점수만 포함했고, 실제 원본 코드와 실행기는 빠져 있었다.

| 구분 | 위치 | 고정 commit / 상태 |
|---|---|---|
| RUDDER 원저자 전체 저장소 | `author/RUDDER` Git submodule | `bd149d8dbde6a9097dd02aea80f83d0bd8876761` |
| GrAInS 원저자 전체 저장소 | `author/GrAInS` Git submodule | `8ad78db3040fbb7b0aec9e544a3dcc7b9c050b5b` |
| RUDDER 100쌍 실행 | `snapshots/rudder_compat/` | 서버에서 사용한 실행기·원본 변경 파일·SHA 기록 |
| GrAInS 추출50 및 Tensor 진단 실행 | `snapshots/grains_checkpoint/` | 서버에서 사용한 실행기·원본 변경 파일·SHA 기록 |
| GrAInS 원본 tuple 계약 후속 | `snapshots/grains_tuple_generation/` | 실행기·사전검사 기록; 생성 시험 결과 미완료 |
| 공통 자체 런타임 | `support/` | 해당 서버에서 보존한 Python helper 사본 |

원저자 코드는 이 저장소에 복사한 vendor 파일이 아니다. Git submodule은 해당 commit의 원본 저장소 전체를 연결한다. 첫 clone 이후 `git submodule update --init --recursive`로 받을 수 있다. `author/`의 코드가 원본이고, `source_changes/`에는 **서버 실행 당시 원본과 달랐던 파일만** 별도로 보존했다. RUDDER는 Qwen import·prompt·고정 모델 경로, GrAInS는 import·키워드·모델 경로 및 checkpoint 메모리 조정이 포함된다. `source_changes/`를 원본 commit 자체로 읽으면 안 된다.

고정 commit의 추적 파일은 RUDDER 17개(그중 Python 13개), GrAInS 26개(그중 Python·shell 19개)다. RUDDER clone에는 별도 LICENSE 파일이 없으므로 코드 전체를 이 저장소의 일반 파일로 재배포하지 않고 원본 commit을 연결했다.

`snapshots/rudder_compat/operations/`와 `snapshots/grains_checkpoint/operations/`는 우리가 만든 래퍼·queue·메모리 관리 코드다. 원저자 CLI만을 직접 실행한 결과라고 표기하지 않는다. `execution_manifest_compat_20261026.json`은 원저자 clone의 모든 추적 파일과 자체 실행기의 실행 당시 SHA-256을 기록한다. `publication_sha256.json`은 이 PR에 복사한 스냅샷과 지원 파일의 바이트 해시다. 두 실행 프로필은 서로 다른 시점의 고정 소스라서 같은 이름의 파일도 덮어 합치지 않았다.
원본 파일의 서버 SHA를 확인할 때 Windows checkout의 줄바꿈 변환이 있으면 작업 파일 대신 고정 commit의 Git blob 바이트를 사용해야 한다.

GrAInS `grains_checkpoint`의 생성 진단은 현재 Transformers의 Tensor 출력을 받는 adapter를 사용했다. 원저자 tuple 정규화 경로는 `grains_tuple_generation/tuple_bridge.py`로 분리했으며 그 평가 점수는 아직 없다. `grains_tuple_generation/run_generation.py`는 `torch.load(weights_only=False)`를 사용하므로 해당 실행에서 SHA·journal로 확인한 신뢰 가능한 산출물에만 적용해야 한다.
원저자 저장소의 VLM 생성 helper는 공개 코드에 없어서 이 후속 실행의 생성 경로는 명시적인 자체 재구성이다.

이 코드 묶음은 **서버에서 사용한 상태의 보존본**이다. `/workspace/...` 절대경로, 고정 Qwen 가중치, COCO 이미지·주석, POPE/SPA-VL 입력, 원본 CHAIR 채점기, 외부 Python 환경과 `PYTHONPATH`가 필요하다. 새 checkout에서 곧바로 실행되는 패키지로 검증하지 않았다. `external_manifest_compat_20261026.json`에 외부 입력·채점기·런타임의 SHA가 있다. 모델·데이터·원문 생성 결과·대형 추출 artifact는 GitHub에 포함하지 않았다. 이 파일들을 공개 코드나 합성 fixture로 대체하면 같은 실행이라고 주장할 수 없다.

선정 점수와 실패 포함 범위는 [선정 기록](../../docs/baseline_selection_20261026.md) 및 [pilot 원자료 요약](../../docs/results/qwen_pilot_20261026.json)을 참조한다. REVIS·VTI는 여기의 두 새 clone이 아니라 각자 검증된 별도 프로필이며, 이 코드 묶음이 네 방법의 전체 재현 완료를 뜻하지 않는다.
