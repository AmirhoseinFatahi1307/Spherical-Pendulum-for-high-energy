"""Spherical pendulum - FastAPI server.   Run:  python app.py   (opens http://127.0.0.1:8000)

All physics (the energy equation, theta_dot of every frame, positions, g-force) is computed in
pendulum.py. This file only serves the page and returns Python's results as JSON.
The browser (index.html) only draws what Python computed.
"""

import math
import threading
import webbrowser
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, PlainTextResponse

from pendulum import Params, simulate, to_csv

BASE = Path(__file__).parent
app = FastAPI(title="Spherical pendulum")


@lru_cache(maxsize=32)
def run_sim(m, R, g, th0, w0, ph0):
    p = Params(m=m, R=R, g=g, theta0=th0, theta_dot0=w0, phi_dot0=ph0)
    return p, simulate(p)  # 30 s at 60 frames/s


@app.get("/api/simulate")
def api_simulate(
    m: float = Query(1.0, gt=0, le=50),
    R: float = Query(1.0, ge=0.1, le=5),
    g: float = Query(9.81, gt=0, le=100),
    th0: float = Query(0.8, ge=0, le=math.pi),
    w0: float = Query(0.0, ge=-50, le=50),
    ph0: float = Query(1.5, ge=-50, le=50),
):
    p, sim = run_sim(m, R, g, th0, w0, ph0)
    data = {
        k: [None if math.isnan(v) else round(v, 6) for v in arr.tolist()]
        for k, arr in sim.items()
    }
    data.update(E=p.energy, p_phi=p.p_phi, high=p.energy > p.m * p.g * p.R, fps=60)
    return data  # NaN (sqrt of a negative number) is sent as null


@app.get("/api/csv")
def api_csv(
    m: float = 1.0,
    R: float = 1.0,
    g: float = 9.81,
    th0: float = 0.8,
    w0: float = 0.0,
    ph0: float = 1.5,
):
    _, sim = run_sim(m, R, g, th0, w0, ph0)
    return PlainTextResponse(
        to_csv(sim),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=theta_dot_frames.csv"},
    )


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


if __name__ == "__main__":
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
