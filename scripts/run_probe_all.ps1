# Run the seen-crop linear probe for all three nested configs against ONE data pool.
#
# WHY THIS EXISTS: probe_seen.py derives its class list from whatever SAGE shards are on disk, so
# running A today and C three weeks ago yields configs that are NOT comparable (probe_seen_C.json
# from 14 Jul 2026 had 166 classes; the pool has grown since). Always run A, B and C back-to-back.
#
# Detached usage (survives the agent's tool timeout):
#   Start-Process powershell -ArgumentList '-NoProfile','-File','scripts\run_probe_all.ps1' `
#       -RedirectStandardOutput logs\probe_all.log -RedirectStandardError logs\probe_all.err -NoNewWindow

$ErrorActionPreference = 'Stop'
$py = "C:\Users\PV Abhiram\.pyenv\pyenv-win\versions\3.11.9\python.exe"
Set-Location "C:\Projects\plant-disease-edge"

foreach ($e in @('B', 'C', 'A')) {
    Write-Output "===== EXP $e  ($(Get-Date -Format o)) ====="
    & $py -u scripts/probe_seen.py --exp $e
    Write-Output "===== EXP $e done (exit $LASTEXITCODE) ====="
}
Write-Output "ALL DONE $(Get-Date -Format o)"
