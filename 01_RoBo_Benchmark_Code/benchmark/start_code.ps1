# Startskript fuer den Benchmark (PowerShell, aus dem Ordner benchmark/ aufrufen).
#
#   .\start_code.ps1 plan                         # Experimentplan erstellen + anzeigen (rechnet nichts)
#   .\start_code.ps1 start                        # main.py im Hintergrund starten (Log in results/raw/)
#   .\start_code.ps1 start -Jobs 4 -Name test     # mit 4 Prozessen, Dateiname results_<Datum>_test.csv
#   .\start_code.ps1 start -Max 3                 # kurzer Testlauf mit 3 Experimenten
#   .\start_code.ps1 status                       # Live-Status (Log mitlesen, Strg+C beendet nur die Anzeige)
#   .\start_code.ps1 stop                         # laufenden Benchmark inkl. Worker-Prozesse beenden
#
# Falls Skripte blockiert sind:  powershell -ExecutionPolicy Bypass -File .\start_code.ps1 start

param(
    [Parameter(Position = 0)]
    [ValidateSet("plan", "start", "stop", "status")]
    [string]$Action = "status",
    [int]$Jobs = 0,         # 0 = config.N_JOBS
    [int]$Max = 0,          # 0 = kompletter Plan
    [string]$Name = ""      # Zusatz im Dateinamen, z.B. "rep10_stableopt"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$RawDir = Join-Path $PSScriptRoot "results\raw"
$StateFile = Join-Path $RawDir ".current_run.json"
$env:PYTHONUNBUFFERED = "1"        # Ausgabe sofort ins Log schreiben
$env:PYTHONIOENCODING = "utf-8"    # Log als UTF-8 (nicht UTF-16 wie bei PowerShell-Umleitung)

function Get-RunState {
    if (-not (Test-Path $StateFile)) { return $null }
    $state = Get-Content $StateFile -Raw | ConvertFrom-Json
    $proc = Get-Process -Id $state.pid -ErrorAction SilentlyContinue
    $state | Add-Member -NotePropertyName running -NotePropertyValue ($null -ne $proc) -Force
    return $state
}

switch ($Action) {

    "plan" {
        $planPath = "results/raw/experiment_plan_$(Get-Date -Format yyyy-MM-dd).csv"
        $code = @"
import pandas as pd
from preprocessing.experiment_design import create_experiment_table, remove_duplicate_nominal_runs
df = pd.DataFrame(remove_duplicate_nominal_runs(create_experiment_table()))
df.to_csv(r'$planPath', index=False)
print(f'{len(df)} Experimente geplant, gespeichert in $planPath')
print()
print(df.groupby(['algorithm', 'scenario']).size().unstack(fill_value=0))
"@
        python -c $code
    }

    "start" {
        $state = Get-RunState
        if ($state -and $state.running) {
            Write-Host "Es laeuft bereits ein Benchmark (PID $($state.pid)). Erst '.\start_code.ps1 stop'." -ForegroundColor Yellow
            return
        }
        New-Item -ItemType Directory -Force $RawDir | Out-Null
        $suffix = if ($Name) { "_$Name" } else { "" }
        $stamp = "$(Get-Date -Format yyyy-MM-dd)$suffix"
        $output = "results/raw/results_$stamp.csv"
        $log = Join-Path $RawDir "run_$stamp.log"
        $errLog = Join-Path $RawDir "run_$stamp.err.log"

        $pyArgs = @("main.py", "--output", $output)
        if ($Jobs -ne 0) { $pyArgs += @("--jobs", $Jobs) }
        if ($Max -gt 0) { $pyArgs += @("--max-experiments", $Max) }

        $proc = Start-Process python -ArgumentList $pyArgs -WorkingDirectory $PSScriptRoot `
            -RedirectStandardOutput $log -RedirectStandardError $errLog -WindowStyle Hidden -PassThru
        @{ pid = $proc.Id; log = $log; errlog = $errLog; output = $output; started = (Get-Date -Format s) } |
            ConvertTo-Json | Set-Content $StateFile -Encoding utf8

        Write-Host "Benchmark gestartet (PID $($proc.Id))." -ForegroundColor Green
        Write-Host "  Ergebnisse: $output"
        Write-Host "  Log:        $log"
        Write-Host "Live-Status: .\start_code.ps1 status   |   Stoppen: .\start_code.ps1 stop"
    }

    "stop" {
        $state = Get-RunState
        if (-not $state -or -not $state.running) {
            Write-Host "Kein laufender Benchmark gefunden."
            return
        }
        # /T beendet auch die Worker-Prozesse (multiprocessing)
        taskkill /PID $state.pid /T /F | Out-Null
        Write-Host "Benchmark (PID $($state.pid)) gestoppt." -ForegroundColor Yellow
        Write-Host "Fertige Experimente stehen in $($state.output); erneutes 'start' mit gleichem -Name setzt dort fort."
    }

    "status" {
        $state = Get-RunState
        if (-not $state) {
            Write-Host "Noch kein Lauf ueber dieses Skript gestartet."
            return
        }
        $label = if ($state.running) { "LAEUFT" } else { "BEENDET" }
        Write-Host "Status: $label  (PID $($state.pid), gestartet $($state.started))" -ForegroundColor Cyan
        Write-Host "Log: $($state.log)"
        if ((Test-Path $state.errlog) -and (Get-Item $state.errlog).Length -gt 0) {
            Write-Host "Achtung, Fehlerausgabe vorhanden: $($state.errlog)" -ForegroundColor Red
        }
        Write-Host "-----"
        if ($state.running) {
            Get-Content $state.log -Tail 20 -Wait    # Strg+C beendet nur die Anzeige, nicht den Benchmark
        } else {
            Get-Content $state.log -Tail 20
        }
    }
}
