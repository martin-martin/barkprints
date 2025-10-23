"""Tests for command-line interface."""

import subprocess
import sys
from pathlib import Path

import pytest


def run_barkprints(args):
    """Run barkprints CLI command and return result."""
    cmd = [sys.executable, "-m", "barkprints"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    return result


def test_cli_help():
    """Test CLI help command."""
    result = run_barkprints(["--help"])
    
    assert result.returncode == 0
    assert "barkprints" in result.stdout
    assert "Generate poetry" in result.stdout


def test_cli_list_vocabularies():
    """Test listing vocabularies."""
    result = run_barkprints(["--list-vocabularies"])
    
    assert result.returncode == 0
    assert "nature" in result.stdout
    assert "news" in result.stdout


def test_cli_list_formats():
    """Test listing output formats."""
    result = run_barkprints(["--list-formats"])
    
    assert result.returncode == 0
    assert "haiku" in result.stdout
    assert "commentary" in result.stdout
    assert "sentence" in result.stdout


def test_cli_with_real_bark_image():
    """Test CLI with the real bark image if available."""
    bark_path = Path("barks.jpg")
    
    if not bark_path.exists():
        pytest.skip("Real bark image not found")
    
    result = run_barkprints(["barks.jpg", "-v", "nature"])
    
    assert result.returncode == 0
    assert len(result.stdout) > 0


def test_cli_with_nonexistent_image():
    """Test CLI with non-existent image."""
    result = run_barkprints(["nonexistent.jpg", "-v", "nature"])
    
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


def test_cli_with_format_override():
    """Test CLI with format override."""
    bark_path = Path("barks.jpg")
    
    if not bark_path.exists():
        pytest.skip("Real bark image not found")
    
    result = run_barkprints(["barks.jpg", "-v", "nature", "-f", "sentence"])
    
    assert result.returncode == 0
    assert len(result.stdout) > 0


def test_cli_determinism():
    """Test that CLI produces deterministic output."""
    bark_path = Path("barks.jpg")
    
    if not bark_path.exists():
        pytest.skip("Real bark image not found")
    
    result1 = run_barkprints(["barks.jpg", "-v", "nature"])
    result2 = run_barkprints(["barks.jpg", "-v", "nature"])
    
    assert result1.returncode == 0
    assert result2.returncode == 0
    assert result1.stdout == result2.stdout


def test_cli_no_arguments():
    """Test CLI with no arguments."""
    result = run_barkprints([])
    
    assert result.returncode != 0
    assert "required" in result.stderr.lower()

