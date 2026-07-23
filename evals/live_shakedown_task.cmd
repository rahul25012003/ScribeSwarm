@echo off
rem One-shot: first real-Gemini shakedown run (scheduled 12:40 IST 2026-07-24,
rem just after the daily quota reset; paced 6.5s/call by llm_min_interval).
cd /d "C:\Users\Rahul\Videos\Mission 2030\ScribeSwarm"
if not exist evals\results mkdir evals\results
.venv\Scripts\python.exe cli.py run "Compare the EV battery supply chains of India, China, and the EU" --thread live-shakedown-v3 > evals\results\live_shakedown_20260724.log 2>&1
schtasks /Delete /TN "ScribeSwarm Live Shakedown" /F
del "%~f0"
