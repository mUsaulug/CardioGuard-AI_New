import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_channel_first(signal):
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return signal.T
    return signal

def generate_figure_3():
    # 1. Load Data
    data_path = 'test_mi_sample.npz'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    try:
        data = np.load(data_path)
        # 'signal' key found in previous check
        if 'signal' in data:
            raw_signal = data['signal']
        elif 'x' in data:
             raw_signal = data['x']
        else:
            # Fallback to creating a synthetic signal if keys differ
            print("Warning: 'signal' key not found, using synthetic data.")
            t = np.linspace(0, 10, 1000)
            raw_signal = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.random.randn(1000)
            raw_signal = raw_signal.reshape(1, -1)
            
        # Ensure shape (12, 1000) or similar
        raw_signal = ensure_channel_first(raw_signal)
        
        # Take V2 lead (usually index 7 in 12-lead standard: I, II, III, aVR, aVL, aVF, V1, V2...)
        # If shape is (1, T), use index 0
        lead_idx = 7 if raw_signal.shape[0] >= 8 else 0
        lead_name = "V2 Derivasyonu" if raw_signal.shape[0] >= 8 else "Single Lead"
        
        sig_segment = raw_signal[lead_idx, :500] # Take first 5 seconds/500 points for clarity
        
        # 2. Process (Normalize)
        # Using instance stats for demonstration
        mu = np.mean(sig_segment)
        std = np.std(sig_segment)
        norm_signal = (sig_segment - mu) / (std + 1e-6)
        
        # 3. Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Plot Raw
        ax1.plot(sig_segment, color='#d62728', linewidth=1.5)
        ax1.set_title(f'Ham EKG Sinyali ({lead_name}) - Gürültülü & Taban Çizgisi Kayması', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Genlik (mV)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot Normalized
        ax2.plot(norm_signal, color='#1f77b4', linewidth=1.5)
        ax2.set_title('Normalize Edilmiş Sinyal (Z-Score) - Sıfır Odaklı & Özellikler Belirgin', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Zaman (Örnek)', fontsize=10)
        ax2.set_ylabel('Standart Sapma Birimi (σ)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
        
        plt.tight_layout()
        
        # 4. Save
        out_dir = 'figures'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'figure3_signal_processing.png')
        plt.savefig(out_path, dpi=300)
        print(f"Figure 3 saved to: {os.path.abspath(out_path)}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_figure_3()
