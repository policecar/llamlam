"""
Test the smoke test itself.

This is a meta-test that ensures the smoke test can run successfully.
"""

import pytest

from llamlam.smoke_test import run_smoke_test


@pytest.mark.slow
def test_smoke_test_passes():
    """Run the smoke test and verify it completes successfully."""
    result = run_smoke_test(verbose=False)
    assert result is True
