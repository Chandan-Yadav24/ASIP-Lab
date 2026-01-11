# run_practicals.py
import os
import streamlit as st

print("ASIP Lab - Streamlit Practicals Launcher")
print("Select practical to run:")
print("1 - FFT Demo")
print("2 - Edge Detection")
print("3 - Morphology")
print("4 - Segmentation")

choice = input("Enter choice (1-4): ")

if choice == "1":
    os.system("streamlit run fft_demo.py")
elif choice == "2":
    os.system("streamlit run edge_detection.py")
elif choice == "3":
    os.system("streamlit run morphology.py")
elif choice == "4":
    os.system("streamlit run segmentation.py")
else:
    print("Invalid choice. Exiting...")
