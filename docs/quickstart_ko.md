# FMBD 빠른 시작 안내

`FMBD`는 SNUPY가 미리 계산한 reduced-order body 데이터를 받아 유연한
다물체 Brownian dynamics를 수행하는 독립 runtime입니다. SNUPY는 MOR
artifact를 만드는 전처리기로만 사용하고, runtime script는 `FMBD`만
import합니다.

## 1. 설치

PyTorch가 포함된 Python 환경에서 프로젝트 root에서 실행합니다.

```bash
python -m pip install -e "FMBD[trajectory]"
```

`trajectory` extra는 PDB/DCD 출력을 위한 MDAnalysis를 설치합니다. 출력이
필요 없다면 `python -m pip install -e FMBD`로 충분합니다.

## 2. MOR artifact 준비

현재 SNUPY MOR pickle을 처음 한 번 변환합니다.

```python
from FMBD import BodyModel

model = BodyModel.from_legacy_mor_data(
    "MOR_RESULTS/example/MOR_data.pkl",
    name="rotor",
    device="cpu",
)
model.save("rotor.bodyrom")
```

이후 simulation에는 `MOR_data.pkl` 대신 안정적인 versioned artifact인
`rotor.bodyrom`을 사용합니다.

## 3. 최소 application 구조

전체 템플릿은 [baseline_two_body.py](../examples/baseline_two_body.py)에
있습니다. 기본 순서는 아래와 같습니다.

```python
model_a = BodyModel.load("body_a.bodyrom", device=device)
model_b = BodyModel.load("body_b.bodyrom", device=device)

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

## 6. Switch 실행

현재 switch application은 project root에서 다음처럼 실행합니다.

```bash
python switch_cycle_new.py
```

설정값·node pair·Mg protocol은 `switch_cycle_new.py` 상단에 있습니다.
결과 PDB, DCD, pickle은 `MOR_RESULTS/SWTICH/switch_cycle_new/`에 생성됩니다.

## 주의 사항

- 모든 body와 interaction은 같은 device/dtype을 사용해야 합니다.
- `dynamic=False`인 body는 interaction과 output에는 참여하지만 적분되지
  않으므로 stator/fixed obstacle에 사용할 수 있습니다.
- full production 실행 전에는 Brownian noise를 끄고 짧은 deterministic
  regression을 먼저 수행하세요.
- package import는 대소문자를 구분합니다: `from FMBD import ...`.
