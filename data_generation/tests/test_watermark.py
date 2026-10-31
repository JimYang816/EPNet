"""Channel selection must preserve actual MAT samples and metadata."""
from io import BytesIO
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

from data_generation.src.config import ROOT
from data_generation.src.pdd import load_pdd_config
from data_generation.src.dataset.compact_protocol import physical_identities
from data_generation.src.channel.watermark import WatermarkArchive


def test_nof1_changes_only_channel_configuration():
    original = load_pdd_config()
    assert original.pop("channel") == "NCS1"
    nof1 = load_pdd_config(ROOT/"configs/data/nof1_pddp.json")
    assert nof1.pop("channel") == "NOF1"
    assert nof1 == original
    assert len(physical_identities(nof1, "test"))*2 == 300


@pytest.mark.parametrize("channel,taps", [("NCS1", 510), ("NOF1", 2048)])
def test_archive_selects_channel_and_keeps_native_axes(tmp_path, channel, taps):
    path = tmp_path/"channels.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, length in (("NCS1", 510), ("NOF1", 2048)):
            payload = BytesIO()
            h = np.full((4, length), (1+2j) if name == "NCS1" else (3-4j))
            savemat(payload, dict(h=h, fs_t=16000/length, fs_tau=16000, fc=14000, V0=.0625))
            archive.writestr(f"Watermark/input/channels/{name}/mat/{name}_001.mat", payload.getvalue())
    archive = WatermarkArchive(path, channel)
    sounding = archive.load(1)
    assert sounding.h.shape == (4, taps)
    np.testing.assert_array_equal(sounding.h, (1+2j) if channel == "NCS1" else (3-4j))
    assert sounding.fs_time == 16000/taps and sounding.fs_delay == 16000
    assert sounding.carrier_hz == 14000 and sounding.v0 == .0625
    assert sounding.duration == 3/sounding.fs_time
    manifest = archive.audit([1])
    assert manifest["channel"] == channel
    assert manifest["records"]["1"]["source"].endswith(f"/{channel}/mat/{channel}_001.mat")
    assert len(sounding.sha256) == len(manifest["archive_sha256"]) == 64
    with pytest.raises(FileNotFoundError, match=f"{channel}_002.mat"):
        archive.load(2)


def test_archive_rejects_unsupported_channel(tmp_path):
    with pytest.raises(ValueError, match="Supported WATERMARK"):
        WatermarkArchive(tmp_path/"channels.zip", "OTHER")
