; Windows installer for Music Remover (Inno Setup 6). Built by .github/workflows/release.yml:
;   python packaging/build.py windows
;   ISCC.exe /DAppVersion=1.0.0 packaging\windows\musicremover.iss   -> dist\MusicRemover-Setup.exe
; Installs per user (no admin rights). PyTorch for the user's GPU and the models are
; downloaded on the first start into %LOCALAPPDATA%\MusicRemover.

#ifndef AppVersion
  #define AppVersion "dev"
#endif
#define BundleDir "..\..\build\windows\MusicRemover"

[Setup]
AppId={{909C4711-8298-41B0-91B8-3410FBE57A2D}
AppName=Music Remover
AppVersion={#AppVersion}
AppPublisher=Music Remover
AppPublisherURL=https://github.com/siba1426/musicremover
DefaultDirName={localappdata}\Programs\MusicRemover
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=MusicRemover-Setup
SetupIconFile=musicremover.ico
UninstallDisplayIcon={app}\musicremover.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Music Remover"; Filename: "{app}\MusicRemover.bat"; WorkingDir: "{app}"; IconFilename: "{app}\musicremover.ico"
Name: "{autodesktop}\Music Remover"; Filename: "{app}\MusicRemover.bat"; WorkingDir: "{app}"; IconFilename: "{app}\musicremover.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\MusicRemover.bat"; WorkingDir: "{app}"; Description: "Start Music Remover now (the first start downloads PyTorch for your GPU, 1-3 GB)"; Flags: postinstall nowait shellexec skipifsilent

[UninstallDelete]
; The app folder (incl. compiled Python files) and the downloaded PyTorch, models and extension copy.
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{localappdata}\MusicRemover"
