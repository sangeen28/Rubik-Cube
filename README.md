# Rubik’s Cube Solver (Streamlit)

A Streamlit web app to scramble and solve a Rubik’s Cube using:

- https://rubik-cube-jne2ldq82akvf5cnrm3ukm.streamlit.app/
- **Baseline tabular Q-learning agent** (`Agent.py`)
- **Optional advanced agent** with beam search + heuristic guidance (`agent2.py`)

## Repo contents
- `app.py` — Streamlit entrypoint
- `puzzle.py` — cube state + move operators
- `Agent.py` — baseline agent
- `agent2.py` — advanced agent (optional, but included here)
- `requirements.txt` — Python dependencies for Streamlit Community Cloud

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (get a shareable link)
1. Create a new GitHub repo (public is easiest).
2. Upload all files in this repository (keep the same filenames; **`Agent.py` is case-sensitive** on Streamlit Cloud).
3. Go to **Streamlit Community Cloud** → **New app**.
4. Select:
   - **Repository**: your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy**.

After it deploys, your public link will look like:
`https://<your-app-name>.streamlit.app`

### Notes
- Streamlit Community Cloud installs dependencies from `requirements.txt`.
- If you ever need a specific Python version, choose it in **Advanced settings** when deploying (Streamlit Cloud does not rely on `runtime.txt` for Python version selection).

## License
Add a license if you plan to open-source this project (MIT is common).
