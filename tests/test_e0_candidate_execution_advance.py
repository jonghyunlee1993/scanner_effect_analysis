from pathlib import Path

import pytest

from advance_e0_candidate_execution_plan import advance_mapping, trial_number


def test_advance_keeps_passing_slots_and_never_reuses_rejected_ranks():
    advanced, rejected = advance_mapping({2: 0, 7: 1}, [2, 9])
    assert advanced == {2: 2, 7: 1, 9: 3}
    assert rejected == {2: 0, 9: None}


def test_advance_rejects_empty_or_duplicate_failures():
    with pytest.raises(ValueError, match="without failed"):
        advance_mapping({2: 0}, [])
    with pytest.raises(ValueError, match="must be unique"):
        advance_mapping({2: 0}, [2, 2])


def test_trial_number_requires_versioned_directory():
    assert trial_number(Path("s/trial_009")) == 9
    with pytest.raises(ValueError, match="not versioned"):
        trial_number(Path("s/latest"))
