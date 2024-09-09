# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:01:01 2024

@author: rijan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import gaussian_filter1d
import os



def load_data(filepath):
    """
    Load the experimental data from a text file.
    Assumes two columns: Raman shift (cm⁻¹) and intensity.
    """
    data = np.loadtxt(filepath)
    raman_shift = data[:, 0]
    intensity = data[:, 1]
    return raman_shift, intensity

def separate_stokes_antistokes(raman_shift, intensity):
    """
    Separate Stokes and anti-Stokes regions based on the sign of Raman shift.
    """
    stokes_indices = raman_shift >= 0
    antistokes_indices = raman_shift < 0

    stokes_shift = raman_shift[stokes_indices]
    stokes_intensity = intensity[stokes_indices]

    antistokes_shift = raman_shift[antistokes_indices]
    antistokes_intensity = intensity[antistokes_indices]

    return stokes_shift, stokes_intensity, antistokes_shift, antistokes_intensity

def normalize_data(intensity):
    """
    Normalize intensity data to the range [0, 1].
    """
    return (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

def delete_files(directory, keywords):
    """Delete files containing specified keywords in the given directory."""
    for filename in os.listdir(directory):
        if any(keyword in filename for keyword in keywords):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

def find_peaks_with_relative_height_filter(raman_shift, intensity, min_distance=4, prominence=0.04, width_threshold=5, height_threshold=0.08, sigma=2, plot=False):
    """
    Find significant peaks using Gaussian smoothing, prominence, width thresholds, and relative height filtering.
    """
    smoothed_intensity = gaussian_filter1d(intensity, sigma=sigma)
    normalized_intensity = normalize_data(smoothed_intensity)
    
    peaks, properties = find_peaks(normalized_intensity, prominence=prominence, distance=min_distance)
    widths = peak_widths(normalized_intensity, peaks, rel_height=0.5)[0]
    width_filtered_peaks = peaks[widths >= width_threshold]
    peak_heights = normalized_intensity[width_filtered_peaks]
    relative_heights = peak_heights / np.max(normalized_intensity)
    filtered_peaks = width_filtered_peaks[relative_heights >= height_threshold]
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(raman_shift, normalized_intensity, label='Normalized Intensity (Smoothed)')
        plt.plot(raman_shift[filtered_peaks], normalized_intensity[filtered_peaks], 'x', label='Filtered Peaks')
        plt.title('Detected Peaks after Applying Prominence, Width, and Relative Height Thresholds')
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Normalized Intensity')
        plt.legend()
        plt.show()

    return raman_shift[filtered_peaks], normalized_intensity[filtered_peaks]

def match_peaks(stokes_peaks, anti_stokes_peaks, threshold=2):
    """
    Match Stokes and anti-Stokes peaks within a given threshold.
    """
    matched_stokes = []
    matched_anti_stokes = []

    for s_peak in stokes_peaks:
        for a_peak in anti_stokes_peaks:
            if abs(s_peak + a_peak) <= threshold:
                matched_stokes.append(s_peak)
                matched_anti_stokes.append(a_peak)
                break

    return matched_stokes, matched_anti_stokes

def calculate_shift(matched_stokes, matched_anti_stokes):
    """
    Calculate the average shift from matched peaks.
    """
    averages = [(st + ast) / 2 for st, ast in zip(matched_stokes, matched_anti_stokes)]
    overall_average = np.mean(averages)
    return overall_average

def shift_data(data, shift):
    """
    Shift the Raman data by the calculated shift.
    """
    shifted_data = data.copy()
    shifted_data[:, 0] -= shift
    return shifted_data

def remove_noise_peaks(peaks, intensity, threshold=0.1):
    """
    Remove noise peaks based on a minimum intensity threshold.
    """
    real_peaks = peaks[intensity >= threshold]
    return real_peaks

def apply_gaussian_broadening(raman_shift, intensity, sigma=1):
    """
    Apply Gaussian broadening to the intensity data.
    """
    return gaussian_filter1d(intensity, sigma=sigma)

def export_data(filename, data):
    """
    Export data to a text file.
    """
    np.savetxt(filename, data, delimiter='\t', fmt='%.6f')

def plot_data(original_data, shifted_data, cleaned_data, broadened_data, filename):
    """
    Plot original, shifted, noise-removed, and Gaussian-broadened data.
    """
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(original_data[0], original_data[1], label='Original Data')
    plt.plot(shifted_data[0], shifted_data[1], label='Shifted Data')
    plt.title('Original and Shifted Data')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(original_data[0], original_data[1], label='Original Data')
    plt.plot(cleaned_data[0], cleaned_data[1], label='Noise-Removed Data')
    plt.plot(broadened_data[0], broadened_data[1], label='Gaussian Broadened Data')
    plt.title('Noise-Removed and Gaussian-Broadened Data')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    directory = 'D:/HfTe5 data centering'
    delete_files(directory, keywords=['broadened', 'cleaned','centered'])
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    print(f"Processing {len(files)} files")

    for file in files:
        filepath = os.path.join(directory, file)
        print(f"Processing file: {file}")
        
        raman_shift, intensity = load_data(filepath)
        stokes_shift, stokes_intensity, antistokes_shift, antistokes_intensity = separate_stokes_antistokes(raman_shift, intensity)
        
        stokes_peaks_shift, stokes_peaks_intensity = find_peaks_with_relative_height_filter(stokes_shift, stokes_intensity, plot=True)
        antistokes_peaks_shift, antistokes_peaks_intensity = find_peaks_with_relative_height_filter(antistokes_shift, antistokes_intensity, plot=True)

        print(f"Detected Stokes peaks at Raman shifts (cm⁻¹): {stokes_peaks_shift}")
        print(f"Detected anti-Stokes peaks at Raman shifts (cm⁻¹): {antistokes_peaks_shift}")

        matched_stokes, matched_anti_stokes = match_peaks(stokes_peaks_shift, antistokes_peaks_shift)
        print(f"Matched Stokes peaks: {matched_stokes}")
        print(f"Matched anti-Stokes peaks: {matched_anti_stokes}")

        if matched_stokes and matched_anti_stokes:
            shift = calculate_shift(matched_stokes, matched_anti_stokes)
            print(f"Calculated shift: {shift}")
        else:
            print("No matched peaks found. Skipping shift calculation.")
            continue

        # Apply the shift to the data
        data = np.column_stack((raman_shift, intensity))
        shifted_data = shift_data(data, shift)

        # Remove noise peaks
        real_stokes_peaks = remove_noise_peaks(stokes_peaks_shift, stokes_peaks_intensity)
        real_antistokes_peaks = remove_noise_peaks(antistokes_peaks_shift, antistokes_peaks_intensity)
        cleaned_intensity = np.zeros_like(intensity)
        cleaned_intensity[np.isin(raman_shift, np.concatenate([real_stokes_peaks, real_antistokes_peaks]))] = intensity[np.isin(raman_shift, np.concatenate([real_stokes_peaks, real_antistokes_peaks]))]

        # Apply Gaussian broadening to cleaned data
        broadened_intensity = apply_gaussian_broadening(raman_shift, cleaned_intensity)

        # Export data
        centered_data_filename = os.path.join(directory, file.replace('.txt', '_centered.txt'))
        export_data(centered_data_filename, shifted_data)

        cleaned_data_filename = os.path.join(directory, file.replace('.txt', '_cleaned.txt'))
        export_data(cleaned_data_filename, np.column_stack((shifted_data[:, 0], cleaned_intensity)))

        broadened_data_filename = os.path.join(directory, file.replace('.txt', '_broadened.txt'))
        export_data(broadened_data_filename, np.column_stack((shifted_data[:, 0], broadened_intensity)))

        # Plot data
        plot_data((raman_shift, intensity), (shifted_data[:, 0], shifted_data[:, 1]), (shifted_data[:, 0], cleaned_intensity), (shifted_data[:, 0], broadened_intensity), file)

if __name__ == "__main__":
    main()
