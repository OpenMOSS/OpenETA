from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sim.bench_worker import _libero_task_diagnostics


class _State:
    def __init__(self, *, contact: bool, contained: bool = False) -> None:
        self.contact = contact
        self.contained = contained
        self.contact_calls: list[object] = []
        self.contain_calls: list[object] = []

    def check_contact(self, other: object) -> bool:
        self.contact_calls.append(other)
        return self.contact

    def check_contain(self, other: object) -> bool:
        self.contain_calls.append(other)
        return self.contained


def _env(predicate: str, first_state: _State, second_state: _State):
    return SimpleNamespace(
        parsed_problem={"goal_state": [[predicate, "object", "target"]]},
        object_states_dict={
            "object": first_state,
            "target": second_state,
        },
        obj_body_id={"object": 0, "target": 1},
        sim=SimpleNamespace(
            data=SimpleNamespace(
                body_xpos=np.asarray(
                    [
                        [0.04, 0.01, 0.91],
                        [0.01, 0.01, 0.90],
                    ],
                    dtype=np.float64,
                )
            )
        ),
        _eval_predicate=lambda _state: False,
    )


def test_on_diagnostics_use_support_to_object_contact_and_z_order() -> None:
    object_state = _State(contact=False)
    support_state = _State(contact=True)

    result = _libero_task_diagnostics(_env("On", object_state, support_state))

    predicate = result["predicates"][0]
    assert predicate["contact"] is True
    assert predicate["support_z_lte_object_z"] is True
    assert "first_z_lte_second_z" not in predicate
    assert support_state.contact_calls == [object_state]
    assert object_state.contact_calls == []


def test_in_diagnostics_use_target_containment_without_contact_error() -> None:
    object_state = _State(contact=False)
    region_state = _State(contact=True, contained=False)

    result = _libero_task_diagnostics(_env("In", object_state, region_state))

    predicate = result["predicates"][0]
    assert predicate["contact"] is True
    assert predicate["contained"] is False
    assert "contact_error" not in predicate
    assert region_state.contact_calls == [object_state]
    assert region_state.contain_calls == [object_state]
