"""Tests for keystroke_auth.data.loader."""

import pandas as pd
import pytest

from keystroke_auth.data.loader import download_dataset, load_dataset


def test_load_dataset_raises_when_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(missing_path)


def test_load_dataset_reads_existing_csv(tmp_path, synthetic_df):
    csv_path = tmp_path / "fake_cmu.csv"
    synthetic_df.to_csv(csv_path, index=False)

    loaded = load_dataset(csv_path)

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == synthetic_df.shape
    assert "subject" in loaded.columns


def test_download_dataset_skips_if_file_exists(tmp_path, monkeypatch):
    existing = tmp_path / "already_here.csv"
    existing.write_text("subject,sessionIndex,rep\n")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("urlretrieve should not be called when file exists")

    monkeypatch.setattr("urllib.request.urlretrieve", fail_if_called)

    result_path = download_dataset(url="http://example.com/fake.csv", filepath=existing)
    assert result_path == existing


def test_download_dataset_wraps_failures_in_runtime_error(tmp_path, monkeypatch):
    target = tmp_path / "new_file.csv"

    def raise_oserror(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr("urllib.request.urlretrieve", raise_oserror)

    with pytest.raises(RuntimeError, match="Download failed"):
        download_dataset(url="http://example.com/fake.csv", filepath=target)
