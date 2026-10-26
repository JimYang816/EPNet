"""CPU audit for one genuine WATERMARK window; does not fit on test truth."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from data_generation.src.pdd import load_pdd_config
from data_generation.src.channel.watermark import WatermarkArchive
from data_generation.src.receiver.chain import ReceiverChain
from data_generation.src.transmitter.ofdm import make_frame, overhead


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watermark", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--allow-model-mismatch", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    with threadpool_limits(4):
        c = load_pdd_config(args.config)
        if args.allow_model_mismatch:
            c["model_mismatch_policy"] = "diagnose"
        sounding = WatermarkArchive(args.watermark, c.get("channel", "NCS1")).load(1)
        chain = ReceiverChain(c)
        clean, y, taps, ref, diagnostic = chain.frame(sounding, 2, make_frame(c, 1, 2))
        noise, variance, power = chain.noise(clean, 1, 2)
        diagnostic.update(metadata=sounding.metadata(), overhead=overhead(c),
                          independent_noise_validation=chain.validate_noise(),
                          support_residual=float(np.sum(abs(taps[..., c["delay_length"]:])**2)/np.sum(abs(taps)**2)),
                          rf_noise_variances=variance.tolist(), signal_power=power,
                          passband_waveform_saved=False, complete=True)
        text = json.dumps(diagnostic, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
