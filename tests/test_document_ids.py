"""Tests for mjswan.document.ids: the slug a name becomes, and scoped unique ids.

Layer: L1 (pure Python).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mjswan.document.ids import name2id, unique_id

# The frontend's `sanitizeName` reads the same table (src/manifest/index.test.ts): a URL
# is resolved by that function and a directory named by this one, so a case they disagree
# on is a link that opens the wrong scene (ADR 0006 §4).
NAME2ID_CASES = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "src/mjswan/template/src/manifest/name2id_cases.json"
    ).read_text()
)


# ===========================================================================
# L1: name2id
# ===========================================================================
class TestName2Id:
    def test_spaces_become_underscores(self):
        assert name2id("My Project") == "my_project"

    def test_hyphens_become_underscores(self):
        assert name2id("Test-Scene") == "test_scene"

    def test_uppercase_becomes_lowercase(self):
        assert name2id("FooBar") == "foobar"

    def test_mixed_separators(self):
        assert name2id("Complex Name-With Spaces") == "complex_name_with_spaces"

    def test_already_clean_unchanged(self):
        assert name2id("simple") == "simple"

    @pytest.mark.parametrize(("name", "expected"), NAME2ID_CASES)
    def test_agrees_with_the_frontend_on_the_shared_table(self, name, expected):
        assert name2id(name) == expected


# ===========================================================================
# L1: unique_id
# ===========================================================================
class TestUniqueId:
    def test_free_base_is_returned_as_is(self):
        assert unique_id("flat_terrain", set()) == "flat_terrain"

    def test_a_taken_base_gets_the_first_free_suffix(self):
        assert unique_id("flat_terrain", {"flat_terrain"}) == "flat_terrain_1"
        assert (
            unique_id("flat_terrain", {"flat_terrain", "flat_terrain_1"})
            == "flat_terrain_2"
        )

    def test_a_gap_in_the_suffixes_is_filled_first(self):
        assert (
            unique_id("s", {"s", "s_2"}) == "s_1"
        )  # `_1` is free, so it is taken before `_3`
