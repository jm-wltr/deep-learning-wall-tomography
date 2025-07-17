@echo off
REM -----------------------------------------------------------------
REM run_all.bat
REM Loops through START_INDEX to END_INDEX, y=0..4, updates basedir.txt and runs COMSOL batch
REM -----------------------------------------------------------------

REM 1) Set paths
set COMSOL_BATCH="C:\Program Files\COMSOL\COMSOL60\Multiphysics\bin\win64\comsolbatch.exe"
set CLASS_PATH=C:\Users\Jaime\Documents\deep-learning-wall-tomography\COMSOL\Simulation.class
set BASE_ROOT=C:\Users\Jaime\Documents\deep-learning-wall-tomography\sections_generator\output\stlsCrop

REM 2) Workspace: where basedir.txt and logs go
set WORK_DIR=C:\Users\Jaime\Documents\deep-learning-wall-tomography\COMSOL
cd /d %WORK_DIR%

REM 3) Loop over folders
setlocal enabledelayedexpansion
set START_INDEX=47
set END_INDEX=124
for /L %%i in (%START_INDEX%,1,%END_INDEX%) do (
    REM Zero-pad index to 5 digits
    set idx=%%i
    set pad=00000!idx!
    for /L %%y in (0,1,4) do (
        set folder=!pad:~-5!_crop%%y
        set dir=%BASE_ROOT%\!folder!

        echo ================================================
        echo Running folder: !dir!
        echo !dir!\> basedir.txt
        echo ===

        REM Run COMSOL batch pointing at basedir.txt
        %COMSOL_BATCH% ^
          -inputfile "%CLASS_PATH%" ^
          -nosave
    )
)
endlocal
echo All simulations launched.