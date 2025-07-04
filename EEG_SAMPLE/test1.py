import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scipy.signal import butter, filtfilt, find_peaks
import warnings
warnings.filterwarnings('ignore')

class MuseBlinkDetector:
    def __init__(self, sample_rate=250):
        self.sample_rate = sample_rate
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_fitted = False
        
    def load_data(self, csv_file):
        """Load Muse 2 EEG data from CSV file"""
        # Read the file and skip header lines
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # Find where actual data starts (after header comments)
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('%'):
                data_start = i
                break
        
        # Read the actual data
        data = pd.read_csv(csv_file, skiprows=data_start, header=None)
        
        # Extract EEG channels (typically columns 1-4 for Muse 2)
        # Column 0 is sample index, columns 1-4 are EEG channels
        eeg_data = data.iloc[:, 1:5].values
        
        print(f"Loaded EEG data shape: {eeg_data.shape}")
        print(f"Duration: {len(eeg_data)/self.sample_rate:.2f} seconds")
        
        return eeg_data
    
    def preprocess_eeg(self, eeg_data):
        """Preprocess EEG data with filtering and artifact removal"""
        processed_data = np.zeros_like(eeg_data)
        
        for ch in range(eeg_data.shape[1]):
            channel_data = eeg_data[:, ch]
            
            # Remove DC offset
            channel_data = channel_data - np.mean(channel_data)
            
            # Bandpass filter (1-50 Hz) to remove baseline drift and high-freq noise
            nyquist = self.sample_rate / 2
            low_freq = 1.0 / nyquist
            high_freq = 50.0 / nyquist
            b, a = butter(4, [low_freq, high_freq], btype='band')
            channel_data = filtfilt(b, a, channel_data)
            
            # Notch filter for 60Hz power line noise
            b_notch, a_notch = signal.iirnotch(60.0, 30, self.sample_rate)
            channel_data = filtfilt(b_notch, a_notch, channel_data)
            
            processed_data[:, ch] = channel_data
        
        return processed_data
    
    def detect_blink_artifacts(self, eeg_data, threshold_factor=3.0):
        """
        Detect eye blinks based on EEG artifact characteristics
        Blinks typically show up as large amplitude artifacts in frontal channels
        """
        # For Muse 2: AF7 and AF8 are frontal channels (indices 1 and 2)
        frontal_channels = eeg_data[:, 1:3]  # AF7 and AF8
        
        # Calculate differential signal (AF7 - AF8) which is sensitive to eye movements
        differential = frontal_channels[:, 0] - frontal_channels[:, 1]
        
        # High-pass filter to emphasize rapid changes (blinks)
        nyquist = self.sample_rate / 2
        high_freq = 1.0 / nyquist
        b, a = butter(4, high_freq, btype='high')
        differential_filtered = filtfilt(b, a, differential)
        
        # Calculate envelope using Hilbert transform
        analytic_signal = signal.hilbert(differential_filtered)
        envelope = np.abs(analytic_signal)
        
        # Smooth the envelope
        window_size = int(0.1 * self.sample_rate)  # 100ms window
        envelope_smooth = signal.savgol_filter(envelope, window_size, 3)
        
        # Detect peaks (potential blinks)
        threshold = np.mean(envelope_smooth) + threshold_factor * np.std(envelope_smooth)
        peaks, properties = find_peaks(envelope_smooth, 
                                     height=threshold,
                                     distance=int(0.2 * self.sample_rate))  # Min 200ms between blinks
        
        return peaks, envelope_smooth, threshold
    
    def extract_features(self, eeg_data, window_size=0.5):
        """Extract features for blink detection"""
        window_samples = int(window_size * self.sample_rate)
        hop_size = window_samples // 4  # 75% overlap
        
        features = []
        labels = []
        
        # Detect blink locations first
        blink_peaks, envelope, threshold = self.detect_blink_artifacts(eeg_data)
        
        # Create binary labels for blink windows
        blink_mask = np.zeros(len(eeg_data))
        for peak in blink_peaks:
            start = max(0, peak - window_samples//4)
            end = min(len(eeg_data), peak + window_samples//4)
            blink_mask[start:end] = 1
        
        # Extract features from sliding windows
        for start_idx in range(0, len(eeg_data) - window_samples, hop_size):
            end_idx = start_idx + window_samples
            window_data = eeg_data[start_idx:end_idx]
            
            # Extract various features
            feature_vector = []
            
            for ch in range(window_data.shape[1]):
                channel_data = window_data[:, ch]
                
                # Statistical features
                feature_vector.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.var(channel_data),
                    np.max(channel_data),
                    np.min(channel_data),
                    np.max(channel_data) - np.min(channel_data),  # Peak-to-peak
                    np.mean(np.abs(channel_data)),  # Mean absolute value
                ])
                
                # Frequency domain features
                freqs, psd = signal.welch(channel_data, self.sample_rate, nperseg=len(channel_data)//4)
                
                # Power in different frequency bands
                delta_power = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
                theta_power = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
                alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 12)])
                beta_power = np.mean(psd[(freqs >= 12) & (freqs <= 30)])
                
                feature_vector.extend([delta_power, theta_power, alpha_power, beta_power])
            
            # Cross-channel features (differential signals)
            # AF7 - AF8 (frontal differential)
            frontal_diff = window_data[:, 1] - window_data[:, 2]
            feature_vector.extend([
                np.mean(frontal_diff),
                np.std(frontal_diff),
                np.max(np.abs(frontal_diff))
            ])
            
            # TP9 - TP10 (temporal differential)
            temporal_diff = window_data[:, 0] - window_data[:, 3]
            feature_vector.extend([
                np.mean(temporal_diff),
                np.std(temporal_diff),
                np.max(np.abs(temporal_diff))
            ])
            
            features.append(feature_vector)
            
            # Label: 1 if majority of window contains blink, 0 otherwise
            window_label = 1 if np.mean(blink_mask[start_idx:end_idx]) > 0.3 else 0
            labels.append(window_label)
        
        return np.array(features), np.array(labels)
    
    def train(self, eeg_data):
        """Train the blink detection model"""
        print("Preprocessing EEG data...")
        processed_data = self.preprocess_eeg(eeg_data)
        
        print("Extracting features...")
        features, labels = self.extract_features(processed_data)
        
        print(f"Feature matrix shape: {features.shape}")
        print(f"Labels distribution: {np.bincount(labels)}")
        
        # Check if we have both classes
        if len(np.unique(labels)) < 2:
            print("Warning: Only one class found. Using synthetic negative examples.")
            # Add some random negative examples
            n_synthetic = len(features) // 3
            synthetic_features = np.random.normal(0, 1, (n_synthetic, features.shape[1]))
            synthetic_labels = np.zeros(n_synthetic)
            
            features = np.vstack([features, synthetic_features])
            labels = np.hstack([labels, synthetic_labels])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        self.is_fitted = True
        return X_test_scaled, y_test, y_pred
    
    def predict_blinks(self, eeg_data):
        """Predict blinks in new EEG data"""
        if not self.is_fitted:
            raise ValueError("Model must be trained first!")
        
        processed_data = self.preprocess_eeg(eeg_data)
        features, _ = self.extract_features(processed_data)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        return predictions, probabilities
    
    def plot_results(self, eeg_data, predictions=None, probabilities=None):
        """Plot EEG data with blink detection results"""
        processed_data = self.preprocess_eeg(eeg_data)
        time_axis = np.arange(len(processed_data)) / self.sample_rate
        
        # Detect blinks for visualization
        blink_peaks, envelope, threshold = self.detect_blink_artifacts(processed_data)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot EEG channels
        channel_names = ['TP9', 'AF7', 'AF8', 'TP10']
        for i, name in enumerate(channel_names):
            axes[0].plot(time_axis, processed_data[:, i] + i*200, label=name)
        
        # Mark detected blinks
        for peak in blink_peaks:
            axes[0].axvline(peak/self.sample_rate, color='red', alpha=0.5, linestyle='--')
        
        axes[0].set_ylabel('Amplitude (μV)')
        axes[0].set_title('EEG Channels with Detected Blinks')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot blink detection envelope
        axes[1].plot(time_axis, envelope, label='Blink Detection Envelope')
        axes[1].axhline(threshold, color='red', linestyle='--', label='Threshold')
        axes[1].scatter(blink_peaks/self.sample_rate, envelope[blink_peaks], 
                       color='red', s=50, zorder=5, label='Detected Blinks')
        axes[1].set_ylabel('Envelope Amplitude')
        axes[1].set_title('Blink Detection Envelope')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot model predictions if available
        if predictions is not None and probabilities is not None:
            window_size = int(0.5 * self.sample_rate)
            hop_size = window_size // 4
            pred_time = np.arange(len(predictions)) * hop_size / self.sample_rate
            
            axes[2].plot(pred_time, probabilities, label='Blink Probability')
            axes[2].fill_between(pred_time, 0, predictions, alpha=0.3, label='Predicted Blinks')
            axes[2].axhline(0.5, color='red', linestyle='--', label='Decision Threshold')
            axes[2].set_ylabel('Probability')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_title('Model Predictions')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Train model to see predictions', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.show()

# Example usage
def main(file_path):
    # Initialize the blink detector
    detector = MuseBlinkDetector(sample_rate=250)
    
    # Load your data
    print("Loading EEG data...")
    eeg_data = detector.load_data(file_path)
    
    # Train the model
    print("Training blink detection model...")
    X_test, y_test, y_pred = detector.train(eeg_data)
    
    # Make predictions on the same data (in practice, use new data)
    print("Making predictions...")
    predictions, probabilities = detector.predict_blinks(eeg_data)
    
    # Plot results
    print("Plotting results...")
    detector.plot_results(eeg_data, predictions, probabilities)
    
    # Print some statistics
    n_blinks = np.sum(predictions)
    duration = len(eeg_data) / detector.sample_rate
    blink_rate = (n_blinks / duration) * 60  # blinks per minute
    
    print(f"\nBlink Detection Summary:")
    print(f"Total recording duration: {duration:.2f} seconds")
    print(f"Detected blinks: {n_blinks}")
    print(f"Average blink rate: {blink_rate:.1f} blinks/minute")

if __name__ == "__main__":
    file = './EEG-VV/S01V_data.csv'
    main(file)
