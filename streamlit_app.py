"""Streamlit Community Cloud entrypoint for MAGI.

Streamlit Community Cloud auto-detects a repo-root ``streamlit_app.py``. The
actual application lives in Module 3 (``module3_decision_report``), which is
installed as the importable ``decision_report`` package via ``requirements.txt``
(``./module3_decision_report[app]``). This wrapper only calls into it, so there
is a single source of truth for the UI.

Local use is unchanged: ``make -C module3_decision_report app`` still runs
``src/decision_report/app.py`` directly.
"""

from decision_report.app import main

main()
