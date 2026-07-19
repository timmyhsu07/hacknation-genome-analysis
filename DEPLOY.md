# Deploying MAGI (Streamlit Community Cloud)

MAGI's web UI (Module 3) is a **Streamlit** app. Streamlit is a stateful,
long-running WebSocket server, so it does **not** run on serverless hosts like
Vercel/Netlify. Deploy it on a Streamlit-native host. The free, one-click path
is **Streamlit Community Cloud**, which builds directly from this GitHub repo.

## What's already wired up

The repo root now contains everything Streamlit Community Cloud needs:

| File | Purpose |
|---|---|
| [`streamlit_app.py`](streamlit_app.py) | Entrypoint Cloud auto-detects; calls `decision_report.app.main()`. |
| [`requirements.txt`](requirements.txt) | Installs the Module 3 package (`./module3_decision_report[app]`) so `decision_report` is importable, plus Streamlit. |
| [`.streamlit/config.toml`](.streamlit/config.toml) | MAGI light theme, applied at the repo root where Cloud runs. |

## Deploy it (one time, ~2 minutes)

This step needs a browser login to your own GitHub/Streamlit account, so it
can't be automated from here — do it once and every future `git push` to the
deployed branch redeploys automatically.

1. Push this branch to GitHub (see below).
2. Go to <https://share.streamlit.io> and sign in with the GitHub account that
   owns `timmyhsu07/MAGI-Microbial-Analysis-for-Genomic-Inhibitors`.
3. Click **Create app → Deploy a public app from GitHub** and set:
   - **Repository**: `timmyhsu07/MAGI-Microbial-Analysis-for-Genomic-Inhibitors`
   - **Branch**: `main` (or whichever branch you push)
   - **Main file path**: `streamlit_app.py`
   - **Advanced settings → Python version**: `3.11`
4. Click **Deploy**. First build installs the deps and boots the app.

## What runs in the cloud

Only the **demonstration (mock) pipeline** works on Community Cloud — it needs
zero real artifacts and exercises every decision branch. The **real pipeline**
tab requires a real Module 1 output directory plus trained Module 2 `.joblib`
models, which aren't shipped in the repo, so it stays inert on Cloud (as the UI
labels it). To demo the real pipeline, run locally:
`make -C module3_decision_report app`.

## Alternatives (also valid Streamlit hosts)

If you'd rather self-host or want a custom domain, the same `streamlit_app.py` +
`requirements.txt` deploy on **Render**, **Railway**, or **Fly.io** with the
start command:

```bash
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```
