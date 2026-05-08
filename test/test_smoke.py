from quad.alignment.base import Align
from quad.normalization.base import Onto
from quad.extraction.base import Reader
from quad.scoring.base import Score


def test_imports() -> None:
    assert Reader
    assert Onto
    assert Align
    assert Score
