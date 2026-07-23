@echo off
rem One-shot: small live benchmark sample - 2 questions x 2 configs, paced
rem (scheduled for 13:35 IST 2026-07-23; resumable if quota cuts it).
cd /d "C:\Users\Rahul\Videos\Mission 2030\ScribeSwarm"
.venv\Scripts\python.exe evals\run_benchmark.py --configs multi_verify,multi_noverify --limit 2 --pace 20 > evals\results\live_benchmark_20260723.log 2>&1
schtasks /Delete /TN "ScribeSwarm Live Benchmark" /F
del "%~f0"
