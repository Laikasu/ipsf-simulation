# iPSF Simulation  

## Overview  
**iPSF Simulation** is a tool designed for simulating the **Interferometric Point Spread Function (iPSF)** that you observe in interferometric scattering (iSCAT) microscopy. It enables researchers and engineers to enter experimental parameters to see how they affect the iPSF.

## Features  
- Simulate PSF and iPSF with configurable parameters
- Efficient computational methods for high-speed simulations

## Usage
- The Main Window displays the generated image. Adjust experimental parameters in the Parameters panel.
- Use the Sweep tab in the Parameters panel to create animations. To visualize the sweep, open the Plot window via View → Show Plot.
- You can export images, animations and plots with File → Save.

## Installation
- For Windows an installer is provided under releases
- For any os the app can be run through python.
```bash
git clone https://github.com/Laikasu/ipsf-simulation.git
cd ./iPSF-Simulation
pip install -r requirements.txt
python main.py

