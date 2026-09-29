"""
Classical DSP-based voice changer.

NO AI voice cloning is used. Two modes are supported:

1. Manual mode: user directly picks a pitch shift (in semitones) and an
   optional formant shift, using phase-vocoder pitch shifting
   (librosa.effects.pitch_shift) -- a signal-processing technique, not a
   generative model.

2. "Match target voice" mode: the user supplies a short reference audio
   clip of the desired character's voice. We analyze classical acoustic
   features of both the source (video's) voice and the target reference
   voice -- specifically the median fundamental frequency (pitch, via
   the YIN algorithm) -- and compute the semitone shift needed to move the
   source's average pitch toward the target's average pitch. This is only
   an approximation (it will not reproduce the target's exact timbre/voice
   identity, since that requires learned voice-conversion/cloning models),
   but it is a fully classical, explainable pitch/tone match.
"""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa
import soundfile as sf


@dataclass
class VoiceProfile:
    median_f0: float          # Hz, median fundamental frequency
    mean_spectral_centroid: float  # Hz, brightness/timbre proxy


def analyze_voice(audio_path: str) -> VoiceProfile:
    """Extract classical acoustic descriptors (pitch, spectral centroid)
    from an audio file using librosa's YIN pitch estimator (a classical
    autocorrelation-based algorithm, not a neural pitch tracker)."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    valid_f0 = f0[np.isfinite(f0) & (f0 > 0)]
    median_f0 = float(np.median(valid_f0)) if valid_f0.size else 0.0

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = float(np.mean(centroid)) if centroid.size else 0.0

    return VoiceProfile(median_f0=median_f0, mean_spectral_centroid=mean_centroid)


def semitone_shift_between(source_profile: VoiceProfile, target_profile: VoiceProfile) -> float:
    """Compute the pitch shift (in semitones) that would move the source
    voice's median pitch to match the target voice's median pitch."""
    if source_profile.median_f0 <= 0 or target_profile.median_f0 <= 0:
        return 0.0
    ratio = target_profile.median_f0 / source_profile.median_f0
    return float(12.0 * np.log2(ratio))


class VoiceChanger:
    """Applies classical pitch-shift / formant-shift DSP to an audio file."""

    def apply_pitch_shift(self, input_audio_path: str, output_audio_path: str,
                           semitones: float) -> str:
        """Shift pitch by `semitones` (positive = higher, negative = lower)
        using a phase-vocoder based algorithm, preserving duration/tempo."""
        y, sr = librosa.load(input_audio_path, sr=None, mono=False)
        if y.ndim == 1:
            shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
        else:
            shifted = np.vstack([
                librosa.effects.pitch_shift(y[ch], sr=sr, n_steps=semitones)
                for ch in range(y.shape[0])
            ])
        sf.write(output_audio_path, shifted.T if shifted.ndim > 1 else shifted, sr)
        return output_audio_path

    def apply_formant_shift(self, input_audio_path: str, output_audio_path: str,
                             formant_ratio: float) -> str:
        """Shift formants (vocal-tract resonance / timbre) independently of
        pitch, via a resample-then-time-stretch trick:
        resampling changes both pitch and formants together; stretching
        back to original duration with a phase vocoder restores the
        original tempo/pitch contour while keeping the shifted formants.
        `formant_ratio` > 1.0 -> shorter vocal tract feel (higher, thinner);
        < 1.0 -> longer vocal tract feel (deeper, richer).
        """
        y, sr = librosa.load(input_audio_path, sr=None, mono=True)
        if formant_ratio <= 0:
            formant_ratio = 1.0

        resampled = librosa.resample(y, orig_sr=sr, target_sr=int(sr * formant_ratio))
        # Time-stretch back to the original number of samples/duration.
        stretch_rate = len(resampled) / max(len(y), 1)
        restretched = librosa.effects.time_stretch(resampled, rate=stretch_rate)

        # Guard against off-by-a-few-samples length mismatch after stretching.
        if len(restretched) < len(y):
            restretched = np.pad(restretched, (0, len(y) - len(restretched)))
        else:
            restretched = restretched[: len(y)]

        sf.write(output_audio_path, restretched, sr)
        return output_audio_path

    def apply_pitch_and_formant(self, input_audio_path: str, output_audio_path: str,
                                 semitones: float, formant_ratio: float = 1.0) -> str:
        """Convenience: apply formant shift first, then pitch shift, in a
        single call, writing only the final result to disk."""
        if abs(formant_ratio - 1.0) < 1e-3:
            return self.apply_pitch_shift(input_audio_path, output_audio_path, semitones)

        tmp_path = output_audio_path + ".formant_tmp.wav"
        self.apply_formant_shift(input_audio_path, tmp_path, formant_ratio)
        try:
            self.apply_pitch_shift(tmp_path, output_audio_path, semitones)
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        return output_audio_path

    def match_target_voice(self, input_audio_path: str, target_reference_audio_path: str,
                            output_audio_path: str,
                            formant_ratio: Optional[float] = None) -> str:
        """Analyze both the source and target reference voices classically
        and shift the source's pitch to approximate the target's average
        pitch. Optionally also apply a manual formant_ratio for extra
        timbre adjustment. This APPROXIMATES tone; it does not clone the
        target's exact voice identity (that requires AI voice conversion,
        intentionally excluded here).
        """
        source_profile = analyze_voice(input_audio_path)
        target_profile = analyze_voice(target_reference_audio_path)
        semitones = semitone_shift_between(source_profile, target_profile)

        effective_formant_ratio = formant_ratio if formant_ratio is not None else 1.0
        return self.apply_pitch_and_formant(
            input_audio_path, output_audio_path, semitones, effective_formant_ratio)
