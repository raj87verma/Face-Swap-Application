@echo off
REM Build a single-file, fully self-contained Windows executable for
REM FaceSwap Studio using PyInstaller. The resulting .exe bundles ALL
REM Python dependencies, the classical facial-landmark model, and
REM ffmpeg.exe -- the end user just double-clicks it, nothing else to
REM install.
REM
REM Run this from the project root (Face-Swap-Application\) inside your
REM activated virtual environment, after: pip install -r requirements.txt

pip install pyinstaller

if not exist "models\shape_predictor_68_face_landmarks.dat" (
    echo Landmark model not found -- downloading it first...
    python scripts\download_landmark_model.py
)

if not exist "ffmpeg.exe" (
    echo.
    echo WARNING: ffmpeg.exe not found in the project root.
    echo Download a Windows static build from https://www.gyan.dev/ffmpeg/builds/
    echo ^(the "essentials" build^), extract it, and copy ffmpeg.exe into this
    echo folder before building, so it gets bundled into the .exe.
    echo Continuing build without it -- audio muxing will fail at runtime
    echo unless ffmpeg is separately available on the target machine's PATH.
    echo.
    pause
)

set ADD_BINARY_ARG=
if exist "ffmpeg.exe" set ADD_BINARY_ARG=--add-binary "ffmpeg.exe;."

pyinstaller ^
    --name "FaceSwapStudio" ^
    --onefile ^
    --windowed ^
    --add-data "models\shape_predictor_68_face_landmarks.dat;models" ^
    %ADD_BINARY_ARG% ^
    --collect-all librosa ^
    --collect-all soundfile ^
    --collect-data dlib ^
    main.py

echo.
echo Build complete. Find the executable in dist\FaceSwapStudio.exe
pause
