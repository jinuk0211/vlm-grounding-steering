# 내일 CHAIR 재개

실험 controller와 생성 worker는 사용자 요청으로 종료됐고 GPU 메모리가 해제됐다. SIGCONT로 재개할 수 없다. 전체 POPE 채점은 완료됐으며 CHAIR 캡션 552/3,000개가 보존됐다. 미완료 작업 4개에 저장된 prefix 16개 응답도 유지한다.

개인 서버 경로·PID·소유자·실행 명령이 포함된 상세 정지 원본과 정확한 재개 명령은 로컬 `outputs/restart-local/PAUSED_CHECKPOINT.json` 및 `PAUSED_HANDOFF_KO.md`에 보관했다. 이 GitHub 요약에는 해당 식별 정보를 포함하지 않는다.

서버에 준비된 CHAIR 재개 스크립트의 CPU 검사 옵션으로 기존 JSONL·claims·완료 마커·소스·벡터 해시를 다시 확인한 뒤 실행 옵션으로 재개한다. 재개 스크립트는 완성된 작업을 건너뛰고 중간 파일에는 검증한 저장 prefix 뒤부터 이어 쓴다. 원래 controller나 초기 startup 오류용 복구 스크립트를 중복 실행하지 않는다.

GPU0 한 장 제한과 기존 protocol·정밀도·문항별 seed를 유지한다. 이번 마감에서는 CPU 검사만 했으며 모델을 다시 띄우지 않았다. CHAIR를 모두 완료한 뒤 공식 채점·독립 검증 결과를 이 결과 PR에 추가한다. 실행 코드 본체는 서버에 보존하고 GitHub에는 코드 해시와 결과를 보관한다.
