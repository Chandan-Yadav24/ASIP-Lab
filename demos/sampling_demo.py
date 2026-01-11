# demos/sampling_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from . import dsp_utils

def run():
    st.subheader("Sampling & Fourier Transforms: The Digital Snapshot")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: How Sampling Changes the Spectrum", expanded=False):
        st.markdown(
            r"""
            **Sampling** is the bridge between the continuous world ($f(t)$) and computers.
            
            ### 1) The Impulse Train Model
            Sampling is mathematically equivalent to multiplying your signal by a set of impulses spaced at $\Delta T$.
            
            ### 2) Spectral Repetition
            In the frequency domain, sampling causes the signal's spectrum to **copy itself infinitely** every $F_s$ Hz.
            - **Over-sampling**: Copies are far apart. Original signal is safe.
            - **Under-sampling (Aliasing)**: Copies overlap! You get "false" frequencies because you can't tell which copy you are looking at.
            
            ### 3) Frequency Resolution vs. Range
            - **Range** (how high you can see): Set by Sampling Rate ($F_s$). Higher $F_s \implies$ Wider range.
            - **Resolution** (how fine you can see): Set by Duration ($T$). Longer record $\implies$ Finer bins.
            """
        )

    st.markdown("---")

    # --- Interactive Demo ----------------------------------------------------
    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Sampling Parameters")
        
        f_sig = st.slider("Signal Frequency (Hz)", 10, 2000, 440, key="samp_f_sig")
        fs = st.slider("Sampling Rate ($F_s$)", 100, 10000, 5000, key="samp_fs")
        duration = st.slider("Record Duration (s)", 0.01, 0.1, 0.02, 0.005, key="samp_dur")
        
        nyquist = fs / 2
        is_aliased = f_sig > nyquist
        
        if is_aliased:
            st.warning(f"⚠️ **Aliasing Active!** Signal ({f_sig}Hz) > Nyquist ({nyquist}Hz)")
            alias_freq = abs(f_sig - round(f_sig/fs)*fs)
            st.write(f"Observed Frequency: **{alias_freq:.1f} Hz**")
        else:
            st.success(f"✅ Healthy Sampling (Signal < Nyquist: {nyquist}Hz)")

        # Audio Generation
        st.write("### 2. Listen to Aliasing")
        # To hear aliasing, we generate at a high rate but sample down
        high_res_fs = 44100
        t_audio = np.linspace(0, 0.5, int(0.5 * high_res_fs))
        audio_sig = np.sin(2 * np.pi * f_sig * t_audio)
        
        # Real-time downsampling for audio
        step = int(high_res_fs / fs)
        aliased_audio = audio_sig[::step]
        aliased_ts = np.arange(len(aliased_audio)) / fs
        
        st.audio(dsp_utils.Wave(aliased_audio, aliased_ts, framerate=fs).get_audio_bytes(), format='audio/wav')
        st.caption(f"Audio at {fs} Hz sampling")

    with col_viz:
        st.write("### 3. Visual Domain Analysis")
        
        # A) Time Domain: Dots on Wave
        t_cont = np.linspace(0, duration, 1000)
        y_cont = np.sin(2 * np.pi * f_sig * t_cont)
        
        n_samples = int(duration * fs)
        t_samp = np.arange(n_samples) / fs
        y_samp = np.sin(2 * np.pi * f_sig * t_samp)
        
        fig_time, ax_time = plt.subplots(figsize=(10, 4))
        ax_time.plot(t_cont, y_cont, label='Continuous Signal', alpha=0.3, color='gray')
        ax_time.scatter(t_samp, y_samp, color='tab:red', s=20, label='Discrete Samples')
        ax_time.stem(t_samp, y_samp, linefmt='r:', markerfmt=' ', basefmt=' ')
        ax_time.set_title("Time Domain: Sampling Snapshot")
        ax_time.set_xlabel("Time (s)")
        ax_time.legend()
        ax_time.grid(alpha=0.2)
        st.pyplot(fig_time)
        
        # B) Frequency Domain: Spectral Copies
        st.write("### 4. Frequency Domain: Periodic Repetition")
        
        # Visualize primary spectrum and first few copies
        # Center = 0, copies at +/- Fs, +/- 2Fs
        f_max_view = 15000
        f_axis = np.linspace(-f_max_view, f_max_view, 2000)
        
        def spectrum_model(f, f_s, signal_f):
            # Model as spikes at centers
            val = np.zeros_like(f)
            for m in range(-3, 4): # 7 copies
                center = m * f_s
                # Each copy has +/- signal_f
                dist = np.minimum(np.abs(f - (center + signal_f)), np.abs(f - (center - signal_f)))
                val += np.exp(-(dist**2) / (2 * (f_s*0.01)**2)) # Gaussian spike
            return val
        
        s_vals = spectrum_model(f_axis, fs, f_sig)
        
        fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
        ax_spec.fill_between(f_axis, s_vals, color='tab:blue', alpha=0.6)
        # Highlight Nyquist zone
        ax_spec.axvspan(-nyquist, nyquist, color='tab:green', alpha=0.1, label='Nyquist Range (Baseband)')
        ax_spec.axvline(nyquist, color='red', linestyle='--', alpha=0.5)
        ax_spec.axvline(-nyquist, color='red', linestyle='--', alpha=0.5)
        
        ax_spec.set_title("Frequency Domain: Infinite Periodic Copies")
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.set_ylabel("Spectral Magnitude")
        ax_spec.legend()
        ax_spec.grid(alpha=0.2)
        st.pyplot(fig_spec)
        
        st.caption("When the blue spikes move outside the green zone, they 'alias' (appear inside the zone from a neighbor copy).")

    # --- Resolution Statistics -----------------------------------------------
    st.markdown("---")
    st.write("### 📏 Resolution & Range Metrics")
    r1, r2, r3 = st.columns(3)
    r1.metric("Max Frequency (Range)", f"{fs/2} Hz")
    r2.metric("Bin Spacing (Resolution)", f"{1/duration:.2f} Hz")
    r3.metric("Total Samples", n_samples)
    
    # Analogy: Strobe
    st.markdown("""
    💡 **Analogy: The Strobe Light & Fan**
    - High-speed strobe = **Oversampling**. You see the blade's true position.
    - Low-speed strobe = **Undersampling**. The blade might appear to spin backwards or stay still (**Aliasing**).
    """)
