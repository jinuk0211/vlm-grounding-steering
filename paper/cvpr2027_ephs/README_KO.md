# EPHS 논문 초안 — CVPR 2027 목표

최종 수정일: 2026-10-23 (Asia/Seoul).

영문 제목: **Evidence-Preserving Head Steering for Context-Robust Object Grounding**

이 문서는 실험 전 연구 초안이다. Abstract, Introduction, Related Work, Method,
Experimental Protocol, 결과표 틀, Limitations, Conclusion과 참고문헌을 포함한다.
성능 숫자, baseline 우월성, GPU 속도 및 메모리 수치는 작성하지 않았다.

## 양식과 편집

- main.tex가 주 문서이고 sections 폴더에서 각 절을 편집한다.
- references.bib에 공식 논문 페이지 및 원문으로 확인한 참고문헌 18개가 있다.
- 익명 review 모드, 2단, 10pt, letterpaper를 사용한다. Paper ID는 미정이다.
- 현재 확인된 공식 author-kit은 CVPR 2026 버전이다. 그 스타일 파일을 수정하지
  않고 사용하며 문서의 confYear만 2027로 설정했다.
- CVPR 2027 Call for Papers는 확인했지만 연결된 AuthorGuidelines는 404였다.
  2027 제출 양식이 완전히 검증된 상태로 해석하면 안 된다.
- 2027 author-kit이 확인되면 공식 cvpr.sty와 ieeenat_fullname.bst 및 지침을
  다시 대조한다. 본문 8쪽 이내를 임시 작업 예산으로 삼되, 2027
  제한은 다시 확인한다. 여백이나 글자 크기를 임의로 줄이지 않는다.

## 논문의 중심 주장 후보

실제 객체의 변화에 반응하는 head를 대조군으로 보정해 선택하고 작은 gain만
학습하면, 실제 객체 인식률을 유지하면서 맥락 때문에 발생하는 환각을 줄일
수 있는가를 검증한다. head 선택, scaling, 객체 제거 평가 자체를 새롭다고
주장하지 않는다.

현재 구현의 배경 가림은 편집 흔적 대조군이다. 함께 등장하는 주변 객체나
배경 의미를 바꾸는 semantic-context 실험은 아직 구현·수행되지 않았으며,
본문에서도 계획으로 구분했다. 원본 activation patching으로 가린 객체에
대한 Yes가 회복되는 것은 oracle 진단이고, 실제 추론 성능 개선이 아니다.

## 실험 후 채워야 할 부분

1. Qwen2.5-VL-7B, Qwen3-VL-8B, Gemma 4 E4B-it의 실제 checkpoint smoke test.
2. 원본 이미지 단위로 분리한 discovery / fit / tune / test 자료.
3. 객체 유무와 의미적 맥락 변화를 분리한 진단 자료 및 편집 품질 확인.
4. Base, threshold calibration, REVIS, RUDDER, GrAInS, VTI의 공통 조건 결과.
5. 가까운 HONES, Seeing or Knowing?, PURGE 및 oDPO와의 비교 범위 확정.
6. 동일 학습 예산의 random/magnitude head 선택, KL/control 제거 ablation.
7. POPE, CHAIR와 객체 recall/길이, Causal-HalBench, MME, 여러 seed 및 신뢰구간.
8. 결과가 뒷받침하는 범위로 Abstract/Introduction/Conclusion의 최종 주장 수정.

Table 2는 forced-choice이고, 자연 생성 답변 및 캡션의 평가는 별도로 보고한다.
FPR에 대응하는 실제 test TPR도 함께 적는다. test 결과를 보고 임계값을 다시
맞추면 안 된다. 빈칸의 대시는 미측정이라는 뜻이며 0점이 아니다.

## 근거와 재현 상태

- EPHS 설명은 현재 src/vlm_grounding/method.py의 실제 수식과 개입 위치를 대조했다.
- CPU 테스트 74개 통과는 구현 검증 결과이며 모델 성능 수치가 아니다.
- 네 steering baseline은 adapted 독립 구현이다. 원 저자 코드 재현으로 표기하지 않는다.
- 참고문헌 작성 중 찾은 oDPO, PURGE, Seeing or Knowing?를 신규성 경계에 반영했다.
- 공개 코드 링크나 저자 정보는 익명 논문 본문에 넣지 않았다.

## 빌드

Overleaf에 소스 ZIP을 올리고 main.tex를 주 문서로 선택하거나, 로컬에서
latexmk -pdf main.tex를 실행한다. Tectonic으로도 컴파일할 수 있다.
초안 상태 표시를 지우는 것만으로 제출 가능한 논문이 되지는 않는다.

완성된 초안은 output/pdf/EPHS_CVPR2027_Draft.pdf이며, 본문 7쪽과 참고문헌
1쪽으로 구성되어 있다. output/EPHS_CVPR2027_Source.zip은 Overleaf용 소스다.
빌드 확인 내용은 BUILD_REPORT.md에 기록했다.
