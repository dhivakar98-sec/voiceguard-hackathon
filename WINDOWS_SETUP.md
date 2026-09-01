# VoiceGuard — Windows Setup Guide (Clone → Run)

Complete steps to run VoiceGuard on Windows, from a fresh machine to a working app,
including the real ML detection model.

> **Use Command Prompt (cmd), not PowerShell.** It avoids a common script-blocking error.
> If you must use PowerShell and activation fails, run `Set-ExecutionPolicy -Scope Process -Bypass` first, or just switch to cmd.

---

## Step 1 — Install Python, ffmpeg, and Git

Open **Command Prompt as Administrator** and paste:

```bat
winget install Python.Python.3.11
winget install Gyan.FFmpeg
winget install Git.Git
```

If `winget` isn't recognized:
- Python → install from https://www.python.org/downloads/ and **tick "Add Python to PATH"** during setup
- Git → https://git-scm.com/download/win
- ffmpeg → `choco install ffmpeg` (needs Chocolatey), or download from https://www.gyan.dev/ffmpeg/builds/ and add it to PATH

---

## Step 2 — Close this terminal and open a NEW one

Open a fresh **Command Prompt** (normal, not admin). This is required so the newly
installed tools are picked up on your PATH.

---

## Step 3 — Verify the installs

```bat
python --version
git --version
```

`python --version` must show **3.10 or higher**. If it says "not recognized,"
reinstall Python with the **"Add Python to PATH"** box ticked.

---

## Step 4 — Clone the project and enter the folder

Go to where you want the project to live, then clone it. **Replace the URL** with your
repo's link (the green **Code** button on the GitHub page gives you this):

```bat
cd C:\Users\%USERNAME%\Downloads
git clone https://github.com/YOURNAME/voiceguard.git
cd voiceguard
```

Everything after this runs from **inside the `voiceguard` folder**.

---

## Step 5 — Create and activate the virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

Your prompt should now start with `(.venv)`. This keeps the project's packages
isolated from the rest of your system.

---

## Step 6 — Install dependencies (core + real ML model)

```bat
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
pip install -r backend\requirements-ml.txt
```

The last line downloads PyTorch and transformers — give it a few minutes.

**If `torch` fails to install**, run this, then re-run the `requirements-ml.txt` line:

```bat
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 7 — Start the app

```bat
python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000
```

Leave this window open — it's the running server.

> **First start downloads the detection model** (a few hundred MB up to ~1 GB) and
> needs internet that one time. After that it's cached and runs offline.

---

## Step 8 — Open it in the browser

Go to **http://localhost:8000** and upload an audio file to test.

---

## Step 9 — Confirm the real ML model loaded

Open **http://localhost:8000/api/health** — it should show `"backend": "ml"`.

If it shows `"heuristic"`, the ML install or the model download didn't finish.
Recheck that Step 6 completed with no errors and that you had internet on the first run.

---

## Everyday use (after first setup)

**Stop the server:** press `Ctrl + C` in the terminal.

**Run it again another day** (skip the install steps):

```bat
cd C:\Users\%USERNAME%\Downloads\voiceguard
.venv\Scripts\activate.bat
python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000
```

**Get the latest code later:**

```bat
git pull
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Reinstall Python with "Add to PATH" ticked; reopen the terminal |
| Activation fails in PowerShell | Use cmd, or run `Set-ExecutionPolicy -Scope Process -Bypass` first |
| `torch` install error | Use the CPU wheel command in Step 6, then re-run the ML requirements |
| `/api/health` shows `heuristic` | ML deps or model download didn't finish — recheck Step 6 + internet |
| mp3/m4a upload errors | ffmpeg isn't installed correctly; use a `.wav` file, or reinstall ffmpeg |
| Port 8000 already in use | Change `--port 8000` to `--port 8080` and open http://localhost:8080 |
| Model won't download (proxy/firewall) | Connect to open internet for the first run; it caches afterward |
