@echo off
setlocal enabledelayedexpansion

REM Define the timeout value
set TIMEOUT=60

echo Running experiment for tsp
python .\parallel_tournament_best2opt.py --filename file-tsp.txt --file_ext tsp --timeout %TIMEOUT%

REM List of file extensions and corresponding filenames
set FILES=kroA100 kroA150 kroA200 kroB100 kroB150 kroB200
REM Loop through each file and run the Python script
for %%f in (%FILES%) do (
    echo Running experiment for %%f...
    python .\parallel_tournament_best2opt.py --filename tsps\%%f.tsp --file_ext %%f --timeout %TIMEOUT%
    echo.
)

echo Running experiment for att48
python .\parallel_tournament_best2opt.py --filename att48.tsp --file_ext att48 --timeout %TIMEOUT%

echo Running experiment for att48
python .\parallel_tournament_best2opt.py --filename d1655.tsp --file_ext d1655 --timeout 1200

echo All experiments completed!
pause