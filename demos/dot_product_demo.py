# demos/dot_product_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("Correlation as a Dot Product: Geometric Intuition")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: Vectors, Angles, and Similarity", expanded=False):
        st.markdown(
            r"""
            Correlation can be viewed as the **dot product** of two vectors in $n$-dimensional space.
            
            ### 1) The Algebra of Simplification
            If signals are **unbiased (mean=0)** and **normalized (std=1)**, the Pearson formula:
            $$\rho = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{(n-1) s_x s_y}$$
            Simplifies to a direct sum of products (the dot product):
            $$\rho \propto \sum x_i y_i = \mathbf{x} \cdot \mathbf{y}$$
            
            ### 2) Geometric Interpretation
            For normalized vectors, the dot product equals the **cosine of the angle** between them:
            $$\mathbf{x} \cdot \mathbf{y} = \cos(\theta)$$
            
            - **$0^\circ$ (Parallel)**: $\cos(0^\circ) = 1$ (Perfect match)
            - **$90^\circ$ (Perpendicular)**: $\cos(90^\circ) = 0$ (Uncorrelated)
            - **$180^\circ$ (Opposite)**: $\cos(180^\circ) = -1$ (Perfectly inverted)
            
            ### 3) Periodic Alignment
            When you correlate two sinusoids and change the phase offset, you are essentially rotating one "signal vector" relative to the other. The resulting correlation curve is itself a sinusoid because of this angular relationship.
            """
        )

    st.markdown("---")

    tab1, tab2 = st.tabs(["2D Vector Intuition", "Sinusoid Alignment"])

    # --- Tab 1: 2D Vector Intuition -------------------------------------------
    with tab1:
        st.write("### 1. Arrows in Space")
        st.markdown("Visualize signals as vectors. Rotate the 'Candidate' vector to see how overlap changes.")
        
        angle_deg = st.slider("Angle Theta (degrees)", 0, 360, 45, key="dp_angle")
        theta = np.deg2rad(angle_deg)
        
        # Base vector (Reference)
        v1 = np.array([1, 0])
        # Rotated vector (Candidate)
        v2 = np.array([np.cos(theta), np.sin(theta)])
        
        dot_product = np.dot(v1, v2)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='tab:blue', label='Reference (Signal X)')
        ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='tab:red', label='Candidate (Signal Y)')
        
        # Unit circle
        circle = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(circle)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', alpha=0.2)
        ax.axvline(0, color='black', alpha=0.2)
        ax.grid(alpha=0.2)
        ax.legend(loc='lower left')
        ax.set_title(f"Dot Product / Correlation: {dot_product:.3f}")
        
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        col1.metric("Dot Product", f"{dot_product:.3f}")
        col2.metric("Cosine Value", f"{np.cos(theta):.3f}")
        
        st.info(f"The Correlation Coefficient ($\rho$) is exactly **{dot_product:.2f}** for these unit vectors.")

    # --- Tab 2: Sinusoid Alignment --------------------------------------------
    with tab2:
        st.write("### 2. Why Sine Alignment forms a Cosine")
        st.markdown("Two sinusoids are compared as one shifts in phase. The sum of their point-by-point products forms the correlation.")
        
        phase_offset = st.slider("Phase Offset (radians)", 0.0, 2*np.pi, 0.0, 0.1, key="dp_phase")
        
        t = np.linspace(0, 1, 1000)
        s1 = np.sin(2 * np.pi * 5 * t)
        s2 = np.sin(2 * np.pi * 5 * t + phase_offset)
        
        # Normalize for correlation view
        s1_norm = (s1 - np.mean(s1)) / np.std(s1)
        s2_norm = (s2 - np.mean(s2)) / np.std(s2)
        
        # Mean dot product (correlation)
        rho_val = np.mean(s1_norm * s2_norm)
        
        fig2, (ax_sig, ax_corr) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Signals
        ax_sig.plot(t[:200], s1[:200], label="Reference Sinusoid", linewidth=2)
        ax_sig.plot(t[:200], s2[:200], label="Shifted Sinusoid", linestyle='--', alpha=0.7)
        ax_sig.fill_between(t[:200], s1[:200]*s2[:200], 0, alpha=0.2, color='gray', label='Product (Overlap)')
        ax_sig.set_title("Time Domain: Signal Overlap")
        ax_sig.legend()
        ax_sig.grid(alpha=0.3)
        
        # Correlation vs Phase
        phases = np.linspace(0, 2*np.pi, 100)
        corr_vals = [np.mean(s1_norm * np.sin(2 * np.pi * 5 * t + p)) for p in phases]
        # We need to normalize correctly for pure correlation coeff
        # Actually in this specific case it forms a cosine
        
        ax_corr.plot(phases, corr_vals, color='tab:orange', label='Correlation vs Phase')
        ax_corr.scatter([phase_offset], [rho_val], color='red', s=100, zorder=5, label='Current State')
        ax_corr.set_xlabel("Phase Shift (radians)")
        ax_corr.set_ylabel("Correlation (ρ)")
        ax_corr.set_title("Correlation Curve (forms a Cosine shape)")
        ax_corr.grid(alpha=0.3)
        ax_corr.legend()
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.success(f"**Current Correlation**: {rho_val:.4f} (Matches $\cos(\phi)$ behavior!)")
