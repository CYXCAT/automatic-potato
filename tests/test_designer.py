from spade_metaworld.complexity import complexity
from spade_metaworld.designer import DesignerConfig, RuleBasedDesigner


def test_designer_generates_strict_progression():
    designer = RuleBasedDesigner(
        DesignerConfig(
            task_families=("reach-v3", "push-v3"),
            initial_goal_variations=2,
            initial_horizon=80,
            horizon_decrement=10,
            goal_variation_increment=2,
            family_every_n_rounds=2,
        ),
        seed=42,
    )
    specs = [designer.propose(i, None, []) for i in range(4)]
    scores = [complexity(spec).total for spec in specs]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores)
    assert specs[2].task_family == "push-v3"
