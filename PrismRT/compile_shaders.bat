@echo off
rem compile_shaders.bat — compile the RT shaders to SPIR-V with glslc.
rem Requires the Vulkan SDK (glslc on PATH). Run from this folder.
setlocal enabledelayedexpansion

where glslc >nul 2>nul
if errorlevel 1 (
    echo [error] glslc not found. Install the Vulkan SDK and add it to PATH.
    exit /b 1
)

set OUT=spv
if not exist "%OUT%" mkdir "%OUT%"

set FLAGS=--target-env=vulkan1.2 -O

for %%S in (prism.rgen photon.rgen prism.rchit prism.rmiss) do (
    echo compiling %%S
    glslc %FLAGS% "%%S" -o "%OUT%\%%S.spv" || exit /b 1
)

echo compiling resolve.comp
glslc %FLAGS% resolve.comp -o "%OUT%\resolve.comp.spv" || exit /b 1

echo.
echo Done. SPIR-V written to %OUT%\
endlocal
