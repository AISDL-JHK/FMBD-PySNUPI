# FMBD-SNUPI 빠른 시작 안내

`FMBD-SNUPI`는 PySNUPI가 미리 계산한 reduced-order body 데이터를 받아
유연한 다물체 Brownian dynamics를 수행하는 독립 runtime입니다. PySNUPI는 MOR
artifact를 만드는 전처리기로만 사용하고, runtime script는 `FMBD`만
import합니다.

배포 패키지 이름은 `FMBD-SNUPI`이고 Python import namespace는 `FMBD`입니다.

## 1. 설치

프로젝트 root에서 실행합니다. 기본 runtime은 CPU와 CUDA 모두 Torch를
사용합니다.

```bash
python -m pip install -e ".[trajectory]"
```

`trajectory` extra는 PDB/DCD 출력을 위한 MDAnalysis를 설치합니다. 출력이
필요 없다면 `python -m pip install -e .`로 충분합니다.

CUDA에서는 서버 드라이버와 CUDA runtime에 맞는 Torch를 먼저 설치한 뒤
FMBD-SNUPI를 설치합니다. 실행 device는 모델을 읽을 때 지정합니다.

```python
model = BodyModel.from_fmbd_data("FMBD_data.pkl", device="cuda:0")
```

중간에 개발했던 NumPy/CuPy backend는 `FMBD.legacy_numpy_cupy` 아래에
보존되어 있습니다. 재현 목적으로 사용할 때만 `legacy-numpy-cupy`,
`legacy-cuda12`, `legacy-cuda13` extra를 설치합니다.

## 2. PySNUPI FMBD artifact 준비

PySNUPI의 FMBD input 예제가 생성한 `FMBD_data.pkl`을 직접 읽습니다.

```python
from FMBD import BodyModel

model = BodyModel.from_fmbd_data("FMBD_data.pkl", device="cpu")
```

반복해서 배포할 때는 `model.save("rotor.bodyrom")`으로 FMBD 자체
versioned artifact를 만들고 이후 `BodyModel.load(...)`로 읽을 수도 있습니다.

## 3. 최소 application 구조

전체 템플릿은 [baseline_two_body.py](../examples/baseline_two_body.py)에
있습니다. 기본 순서는 아래와 같습니다.

```python
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_a = BodyModel.from_fmbd_data("body_a/FMBD_data.pkl", device=device)
model_b = BodyModel.from_fmbd_data("body_b/FMBD_data.pkl", device=device)

system = FMBDSystem()
system.add_body("a", Body(model_a, BodyState.at_reference(model_a)))
system.add_body("b", Body(model_b, BodyState.at_reference(model_b)))

system.add_interaction(DebyeHuckelInteraction("a", "b", charge=0.7))

simulation = Simulation(system, protocol, integrator, observers=[...])
simulation.run()
```

`BodyModel`은 불변 데이터이고, `BodyState(q, R, c)`는 매 step 변하는
modal deformation, rotation, translation입니다. 동일 `BodyModel`로 여러
`Body`를 만들 수 있습니다.

PySNUPI가 생성한 입력은 변환 없이 바로 읽습니다.

```python
model = BodyModel.from_fmbd_data("FMBD_data.pkl", device=device)
```

## 4. Interaction 추가

Morse selected pair:

```python
system.add_interaction(
    MorsePairInteraction("a", "b", pairs, epsilon=42.2, a=2.475, r0=0.3831)
)
```

일반 Debye–Hückel interaction:

```python
system.add_interaction(
    DebyeHuckelInteraction(
        "a", "b",
        charge=0.7,
        concentration_key="Mg_mM",
        cutoff_multiplier=4.0,
        skin=2.0,
        excluded_nodes_a=excluded_a,
        excluded_nodes_b=excluded_b,
    )
)
```

농도는 `PiecewiseProtocol`이 매 step `context.environment["Mg_mM"]`에
넣습니다. Debye–Hückel interaction은 이 값을 이용해 Debye length와
Verlet neighbor list를 자동으로 갱신합니다.

새 물리는 `compute(system, context, diagnostics)`가 `InteractionResult`를
반환하도록 작성한 뒤 `system.add_interaction(...)`으로 넣습니다.

## 5. PDB/DCD 출력

```python
observers = [
    PDBTopologyWriter("output/system.pdb"),
    DCDTrajectoryWriter("output/trajectory.dcd", every=1000),
]
```

PDB는 시작 시 한 번 생성하며 각 body의 internal connectivity와 선택한
`InterBodyBond`를 포함합니다. DCD는 초기 프레임, `every` 간격 프레임,
그리고 마지막 프레임을 기록합니다. 내부 단위는 nm이고, PDB/DCD를 쓸 때만
Å로 변환합니다.

## 6. 적분기 선택

기본값은 중간 형상에서 힘을 다시 계산하는 midpoint Brownian입니다.

```python
integrator = OverdampedFMBDIntegrator(
    dt=5.0,
    options=IntegratorOptions(scheme="midpoint_brownian"),
)
```

기존 Euler Brownian 결과를 재현할 때는
`IntegratorOptions(scheme="euler_brownian")`을 사용합니다. midpoint는
`Simulation`을 통해 실행하면 중간 형상 재구성과 force 재평가가 자동으로
수행됩니다.

## 7. 예제 실행

`examples/baseline_two_body.py`의 두 `FMBD_data.pkl` 경로와 node pair를 실제
입력으로 바꾼 뒤 project root에서 실행합니다.

```bash
python examples/baseline_two_body.py
```

예제는 사용 가능한 CUDA device를 우선 선택하고, 없으면 CPU로 실행합니다.
실행 중 `output/simulation.log`,
`output/system.pdb`, `output/trajectory.dcd`가 순차적으로 기록됩니다.

## 주의 사항

- 모든 body와 interaction은 같은 device/dtype을 사용해야 합니다.
- `dynamic=False`인 body는 interaction과 output에는 참여하지만 적분되지
  않으므로 stator/fixed obstacle에 사용할 수 있습니다.
- full production 실행 전에는 Brownian noise를 끄고 짧은 deterministic
  regression을 먼저 수행하세요.
- package import는 대소문자를 구분합니다: `from FMBD import ...`.
