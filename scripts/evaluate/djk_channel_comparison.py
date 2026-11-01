"""Common-probe channel comparison; never runs an equalizer or bit decoder.

DJK: recovered 24-bit samples in tagged acquisition packets.
WATERMARK: replay the same 50-ms, 6-kHz LFM and central-tone preamble through
the original measured TVIR, retaining its within-probe time variation.
Both: 16-kHz complex baseband, common weighted matched filter and common
delay/gain/phase-invariant comparison. Single-tone frequency fits are diagnostic
only: deep fades can make them unreliable, so no tone-based correction is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
FS = 16000
PERIOD = .75
LAGS = np.arange(-80, 481)  # -5 through +30 ms; common primary window ends at 25 ms.
DELAY_MS = LAGS/16
CHANNELS = ["DJK", "NOF1", "NCS1"]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def quantiles(x):
    return dict(zip(["p05", "median", "p95"], np.quantile(x, [.05, .5, .95]).tolist()))


def probe():
    t = np.arange(800)/FS
    chirp = np.exp(2j*np.pi*(3000*t-60000*t*t))
    preamble = np.zeros(round(.21*FS), complex)
    preamble[:800] = chirp
    preamble[round(.1*FS):round(.186*FS)] = .5
    return chirp, preamble


def replay_tvir(h, fs_time, input_signal, start, output_count):
    """Exact linear TVIR interpolation, evaluated efficiently by stationary convolutions."""
    position = (start+np.arange(output_count)/FS)*fs_time
    index = np.floor(position).astype(int)
    alpha = position-index
    if index.min() < 0 or index.max()+1 >= len(h):
        raise ValueError("Probe leaves the measured TVIR interval")
    convolutions = {int(k): signal.fftconvolve(input_signal, h[k])[:output_count]
                    for k in np.unique(np.r_[index, index+1])}
    result = np.zeros(output_count, complex)
    for k in np.unique(index):
        selected = np.flatnonzero(index == k)
        result[selected] = ((1-alpha[selected])*convolutions[int(k)][selected]
                            + alpha[selected]*convolutions[int(k+1)][selected])
    return result


def matched_observation(y, expected_index, search_before=.002, search_after=.03, compensate=False):
    chirp, _ = probe()
    tone = y[expected_index+round(.108*FS):expected_index+round(.175*FS)]
    t = np.arange(len(tone))/FS
    fit = np.polyfit(t, np.unwrap(np.angle(tone)), 1, w=np.maximum(abs(tone), 1e-20))
    frequency = float(fit[0]/(2*np.pi))
    z = y*np.exp(-2j*np.pi*frequency*np.arange(len(y))/FS) if compensate else y
    corr = signal.correlate(z, chirp*signal.windows.hann(800, sym=False), method="fft", mode="valid")
    lo = max(2, expected_index-round(search_before*FS))
    hi = min(len(corr)-2, expected_index+round(search_after*FS))
    peak = int(lo+np.argmax(abs(corr[lo:hi])))
    power = np.log(np.maximum(abs(corr[peak-1:peak+2])**2, 1e-300))
    denominator = power[0]-2*power[1]+power[2]
    offset = np.clip(.5*(power[0]-power[2])/denominator, -.5, .5) if denominator < 0 else 0.
    return corr, float(peak+offset), frequency


def register_burst(correlations, peak_indices, times, group_id, source, carrier):
    x = np.arange(len(peak_indices))
    # Fit the common linear arrival drift, rather than selecting a new time
    # origin independently for the primary temporal comparison.
    fitted = np.polyval(np.polyfit(x, peak_indices, 1), x)
    responses, peak_registered = [], []
    for corr, center, peak in zip(correlations, fitted, peak_indices):
        interp = CubicSpline(np.arange(len(corr)), corr, extrapolate=False)
        responses.append(interp(center+LAGS))
        peak_registered.append(interp(peak+LAGS))
    return (np.asarray(responses), np.asarray(peak_registered),
            dict(group_id=group_id, source=source, carrier_hz=carrier, times_s=list(map(float,times)),
                 peak_indices=list(map(float,peak_indices)), fitted_peak_indices=fitted.tolist(),
                 residual_arrival_jitter_us=(np.asarray(peak_indices)-fitted).tolist(),
                 fitted_delay_drift_samples_per_probe=float(np.polyfit(x, peak_indices, 1)[0])))


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), git_head=commit)


def write_observations(out, channel, values, registered, groups, frequencies, sources):
    arr = np.asarray(values)
    peak = np.asarray(registered)
    if not np.isfinite(arr).all() or not np.isfinite(peak).all():
        raise ValueError("Invalid response extraction")
    np.savez_compressed(out/f"{channel}_responses.npz", responses=arr, peak_registered=peak,
                        delay_ms=DELAY_MS, common_frequency_hz=np.asarray(frequencies))
    manifest = dict(channel=channel, groups=groups, source_files=sources, **provenance(),
                    parameters=dict(fs_baseband_hz=FS, lfm_duration_ms=50, lfm_bandwidth_hz=6000,
                                    receiver="Hann-weighted matched filter", common_frequency="not corrected; tone fit diagnostic only",
                                    interval_s=PERIOD, primary_window_ms=[-5,25], primary_power_floor_db=-30,
                                    probes_per_burst=11, temporal_delay_search_ms=2.,
                                    primary_registration="per-probe dominant peak for delay metrics; per-pair complex scalar and +/-2 ms delay invariant for temporal metrics"))
    save_json(out/f"{channel}_manifest.json", manifest)


def prepare_djk(out):
    blocks, sources = [], []
    for path in sorted((ROOT/"data/raw/djk").glob("*.raw")):
        data = np.memmap(path, dtype="<u4", mode="r").reshape(-1,8200)
        if not np.array_equal(data[:,0],np.arange(len(data))) or np.any(data[:,1:8]):
            raise ValueError("Unexpected acquisition packet headers")
        payload = data[:,8:].reshape(-1).astype(np.int32)
        if not set(np.unique(payload>>24)).issubset({0x61,0x63}):
            raise ValueError("Unexpected acquisition sample tags")
        samples = ((payload&0xffffff)^0x800000)-0x800000
        blocks.append(samples.astype(np.float32))
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            samples=len(samples),duration_s=len(samples)/96000))
    raw = np.concatenate(blocks)
    mixed = (raw.reshape(-1,4)*np.array([1,-1j,-1,1j],np.complex64)).reshape(-1)
    baseband = 2*signal.resample_poly(mixed,1,6)
    chirp,_ = probe()
    corr = signal.correlate(baseband,chirp,mode="valid",method="fft")
    energy = np.maximum(signal.fftconvolve(abs(baseband)**2,np.ones(800),mode="valid"),1)
    rho = abs(corr)/np.sqrt(800*energy)
    indices,_ = signal.find_peaks(rho,height=.7,prominence=.2,distance=.5*FS)
    bursts = np.split(indices,np.flatnonzero(np.diff(indices)>1.2*FS)+1)
    if len(bursts)!=9 or any(len(b)!=12 for b in bursts):
        raise ValueError(f"Unexpected preamble inventory: {[len(b) for b in bursts]}")
    values, registered, groups, frequencies = [], [], [], []
    for gi,burst in enumerate(bursts):
        # A fixed exact 0.75-s sample grid exposes and estimates common drift.
        anchor=int(burst[0])-round(.015*FS)
        correlations, centers, times, freq=[],[],[],[]
        # Last header lacks the full preamble. Use the same 11-probe burst
        # inventory in every channel, without selecting any decoded outcomes.
        for j in range(11):
            begin=anchor+round(j*PERIOD*FS)
            window=baseband[begin:begin+round(.24*FS)]
            c,p,f=matched_observation(window,round(.015*FS),search_before=.003,search_after=.003)
            correlations.append(c);centers.append(p);times.append((begin+round(.015*FS))/FS);freq.append(f)
        a,b,m=register_burst(correlations,centers,times,gi,"DJK recordings",24000)
        m["residual_arrival_jitter_us"]=(np.asarray(m["residual_arrival_jitter_us"])/FS*1e6).tolist()
        values.append(a);registered.append(b);groups.append(m);frequencies.append(freq)
    write_observations(out,"DJK",values,registered,groups,frequencies,sources)
    print(json.dumps(dict(channel="DJK",bursts=len(groups),preambles=int(np.prod(np.shape(values)[:2])))),flush=True)


def prepare_watermark(out, archive_path, records, starts):
    from data_generation.src.channel.watermark import WatermarkArchive
    _,preamble=probe()
    for ch in ["NOF1","NCS1"]:
        archive=WatermarkArchive(archive_path,ch)
        if records=="all":
            with zipfile.ZipFile(archive_path) as z:
                numbers=sorted(int(Path(n).stem.split("_")[-1]) for n in z.namelist()
                               if f"/{ch}/mat/" in n and n.endswith(".mat"))
        else:
            numbers=[int(v) for v in records.split(",")]
        values,registered,groups,frequencies,sources=[],[],[],[],[]
        for number in numbers:
            sounding=archive.load(number)
            sources.append(sounding.metadata())
            for start in starts:
                correlations,centers,times,freq=[],[],[],[]
                for j in range(11):
                    transmit_time=start+PERIOD*j
                    y=replay_tvir(sounding.h,sounding.fs_time,preamble,transmit_time,round(.225*FS))
                    y=np.pad(y,(round(.015*FS),0))
                    c,p,f=matched_observation(y,round(.015*FS),search_before=0,search_after=.03)
                    correlations.append(c);centers.append(p);times.append(transmit_time);freq.append(f)
                a,b,m=register_burst(correlations,centers,times,len(groups),sounding.source,14000)
                m["record"]=number
                m["residual_arrival_jitter_us"]=(np.asarray(m["residual_arrival_jitter_us"])/FS*1e6).tolist()
                values.append(a);registered.append(b);groups.append(m);frequencies.append(freq)
            if number%10==0 or number==numbers[-1]:
                print(f"{ch}: record {number}, {len(groups)} bursts prepared",flush=True)
        write_observations(out,ch,values,registered,groups,frequencies,sources)


def multipath_metrics(h, delay_ms=DELAY_MS, cutoff_db=-30, window=(-5,25)):
    use=(delay_ms>=window[0])&(delay_ms<=window[1])
    t=delay_ms[use];p=abs(h[...,use])**2
    p=np.where(p>=p.max(axis=-1,keepdims=True)*10**(cutoff_db/10),p,0)
    p=p/p.sum(axis=-1,keepdims=True)
    mean=np.sum(p*t,axis=-1)
    rms=np.sqrt(np.sum(p*(t-mean[...,None])**2,axis=-1))
    outside=np.sum(p[...,abs(t)>1],axis=-1)
    widths=[]
    for row in p.reshape(-1,len(t)):
        a,b=np.interp([.05,.95],np.cumsum(row),t);widths.append(b-a)
    return dict(windowed_rms_delay_ms=rms,energy90_width_ms=np.asarray(widths).reshape(rms.shape),
                energy_outside_main_1ms_fraction=outside),p


def shape_correlation(h, lag, delay_ms=DELAY_MS, window=(-5,25), tail_only=False, delay_search_ms=0):
    use=(delay_ms>=window[0])&(delay_ms<=window[1])
    if tail_only:
        use &= abs(delay_ms)>1
    a=h[:,:-lag,use];b=h[:,lag:,use]
    norm=np.sqrt(np.sum(abs(a)**2,axis=-1)*np.sum(abs(b)**2,axis=-1))
    if delay_search_ms == 0:
        return abs(np.sum(a.conj()*b,axis=-1))/norm
    max_shift=round(delay_search_ms/(delay_ms[1]-delay_ms[0]))
    result=[]
    for first,second,scale in zip(a.reshape(-1,a.shape[-1]),b.reshape(-1,b.shape[-1]),norm.ravel()):
        c=signal.correlate(first,second,mode="full",method="fft")
        center=len(first)-1
        result.append(float(abs(c[center-max_shift:center+max_shift+1]).max()/scale))
    return np.asarray(result).reshape(norm.shape)


def cluster_bootstrap(group_values, cluster_ids, seed=20261101):
    """Median of equal-weight group means; bootstrap independent recording clusters."""
    labels=np.unique(cluster_ids)
    cluster=np.asarray([np.mean(np.asarray(group_values)[np.asarray(cluster_ids)==k]) for k in labels])
    rng=np.random.default_rng(seed)
    draws=cluster[rng.integers(0,len(cluster),(2000,len(cluster)))].mean(axis=1)
    return dict(mean=float(cluster.mean()),cluster_count=len(cluster),
                bootstrap95_interval=np.quantile(draws,[.025,.975]).tolist())


def compare(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"DJK":"#d64d45","NOF1":"#245a87","NCS1":"#239687"}
    raw,manifest,summary,pdps={}, {}, {}, {}
    for ch in CHANNELS:
        raw[ch]=dict(np.load(out/f"{ch}_responses.npz"))
        manifest[ch]=json.loads((out/f"{ch}_manifest.json").read_text(encoding="utf-8"))
        h=raw[ch]["responses"]
        metrics,p=multipath_metrics(raw[ch]["peak_registered"])
        pdps[ch]=p.mean(axis=(0,1))
        cluster=[g.get("record",g["group_id"]) for g in manifest[ch]["groups"]]
        results={k:dict(distribution=quantiles(v.ravel()),
                        cluster_summary=cluster_bootstrap(v.mean(axis=1),cluster)) for k,v in metrics.items()}
        for lag in [1,2,4,8]:
            values=shape_correlation(h,lag,delay_search_ms=2.)
            results[f"shape_correlation_{lag*.75:g}s"]=dict(distribution=quantiles(values.ravel()),
                cluster_summary=cluster_bootstrap(values.mean(axis=1),cluster))
        results["tail_shape_correlation_0.75s"]=dict(distribution=quantiles(shape_correlation(h,1,tail_only=True).ravel()))
        sensitivity=[]
        for threshold in [-20,-25,-30,-35]:
            for end in [15,25,30]:
                m,_=multipath_metrics(raw[ch]["peak_registered"],cutoff_db=threshold,window=(-5,end))
                sensitivity.append(dict(power_cutoff_db=threshold,window_ms=[-5,end],
                                        medians={k:float(np.median(v)) for k,v in m.items()}))
        results["sensitivity"]=sensitivity
        results["temporal_delay_search_sensitivity"]={str(ms):quantiles(shape_correlation(h,1,delay_search_ms=ms).ravel()) for ms in [0.,.5,1.,2.]}
        results["peak_registered_shape_correlation_0.75s"]=quantiles(shape_correlation(raw[ch]["peak_registered"],1).ravel())
        results["bursts"]=h.shape[0];results["probes"]=h.shape[0]*h.shape[1]
        summary[ch]=results
    limitations=[
        "Results compare common-probe effective responses, including the original transducer/recording response, not independently calibrated propagation-only CSI.",
        "DJK uses measured receptions; WATERMARK uses the same LFM replayed through measured TVIR with official linear interpolation. Carrier bands remain 24 and 14 kHz respectively.",
        "Temporal correlation maximizes complex normalized cross-correlation over +/-2 ms relative delay. This removes a common complex gain/phase and timing shift, and can be optimistic for strongly changing multipath. Zero/0.5/1-ms-search sensitivities are retained. This is not raw carrier-phase coherence.",
        "Tone frequency fits are diagnostic only and may be corrupted by deep fades; no single-tone frequency correction is applied to any channel.",
        "DJK probes every 0.75 s. Doppler spectra above +/-0.667 Hz alias; no unaliased Doppler spread or sub-0.75-s wideband coherence is claimed.",
        "Primary delay metrics are windowed to -5..25 ms around the registered dominant arrival. NOF1 late arrivals beyond this window are excluded.",
        "The -30 dB relative response-power cutoff is shared; -20/-25/-35 dB and 15/30 ms window sensitivity results are also retained.",
        "The nine DJK bursts are from one campaign, not nine independent environments. Bootstrap intervals are descriptive cluster variability, not deployment-wide confidence.",
        "No equalizer, BER, decoder, FER, or true-CSI NMSE has been evaluated."]
    report=dict(schema="djk-common-probe-channel-comparison-v1",parameters=manifest["DJK"]["parameters"],
                channels=summary,limitations=limitations,**provenance())
    save_json(out/"comparison.json",report)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    delay=DELAY_MS[(DELAY_MS>=-5)&(DELAY_MS<=25)]
    for ch in CHANNELS:
        axes[0].plot(delay,10*np.log10(pdps[ch]/pdps[ch].max()+1e-9),color=colors[ch],label=ch)
        m,_=multipath_metrics(raw[ch]["peak_registered"])
        values=np.sort(m["windowed_rms_delay_ms"].ravel())
        axes[1].plot(values,np.arange(1,len(values)+1)/len(values),color=colors[ch],label=ch)
        x=np.arange(1,9)*.75
        y=[np.median(shape_correlation(raw[ch]["responses"],lag,delay_search_ms=2.)) for lag in range(1,9)]
        axes[2].plot(x,y,"o-",color=colors[ch],label=ch,markersize=3)
    axes[0].set(xlim=(-2,25),ylim=(-45,1),xlabel="Relative delay (ms)",ylabel="Normalized mean response power (dB)",title="(a) Common-probe delay profile")
    axes[1].set(xlabel="Windowed RMS delay spread (ms)",ylabel="Empirical cumulative fraction",title="(b) Multipath dispersion")
    axes[2].set(ylim=(0,1.02),xlabel="Probe separation (s)",ylabel="Median response-shape correlation",title="(c) Time variation after registration")
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Common 6-kHz / 50-ms LFM analysis | no equalizer evaluation",fontsize=12)
    fig.savefig(out/"common_probe_comparison.png",dpi=220)
    fig.savefig(out/"common_probe_comparison.svg")
    plt.close(fig)
    print(json.dumps({ch:{k:v["distribution"]["median"] for k,v in summary[ch].items() if isinstance(v,dict) and "distribution" in v} for ch in CHANNELS},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["djk","watermark","compare"])
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/runs/djk_common_probe_20261101")
    parser.add_argument("--archive",type=Path,default=ROOT/"data/raw/watermark/compact_cpu_smoke.zip")
    parser.add_argument("--records",default="1,2,3,51,56")
    parser.add_argument("--starts",type=float,nargs="+",default=[2.,12.,22.])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=="djk":
        prepare_djk(args.output)
    elif args.mode=="watermark":
        prepare_watermark(args.output,args.archive,args.records,args.starts)
    else:
        compare(args.output)


if __name__=="__main__":
    main()
