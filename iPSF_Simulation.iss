; ----------------------------------------
; Inno Setup script for IPSF Simulation
; ----------------------------------------

[Setup]
AppName=iPSF Simulation
AppVersion=1.0
DefaultDirName={pf}\iPSF Simulation
DefaultGroupName=iPSF Simulation
OutputDir=output
OutputBaseFilename=iPSF_Simulation_Installer
Compression=lzma
SolidCompression=yes
; Uncomment and set your icon if available
; SetupIconFile=app.ico

[Files]
; Copy entire PyInstaller 'onedir' folder
Source="C:\iPSF-simulation\*"; DestDir="{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
; Start Menu shortcut
Name="{group}\iPSF Simulation"; Filename="{app}\ipsf-simulation.exe"

; Desktop shortcut
Name="{commondesktop}\iPSF Simulation"; Filename="{app}\ipsf-simulation.exe"; Tasks=desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; Flags: unchecked

[Run]
; Launch the app after installation
Filename="{app}\ipsf-simulation.exe"; Description="Launch IPSF Simulation"; Flags=nowait postinstall skipifsilent
