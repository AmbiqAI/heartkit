import math
import os
import random
from typing import Annotated

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..utils import setup_logger
from .defines import AppState, HeartKitState, HKResult

logger = setup_logger(__name__)

g_state = HeartKitState()


def set_global_state(state: HeartKitState):
    """Set global state. NOTE: Add lock and mutate existing object"""
    global g_state  # pylint: disable=global-statement
    g_state = state


def get_global_state() -> HeartKitState:
    """Get global state."""
    return g_state


HeartKitStateDep = Annotated[HeartKitState, Depends(get_global_state)]


def emulate_data():
    """Emulate state updates"""
    logger.info("Emulation: Updating state")
    if random.random() < 0.5:
        return
    num_samples = 2500
    offset = random.uniform(0, 200)
    amp = random.uniform(3, 7)
    data = [
        amp * math.cos((i + offset) * 2 * math.pi / 250) for i in range(num_samples)
    ]
    seg_mask = []
    for _ in range(10):
        seg_mask += (
            50 * [0] + 35 * [1] + 17 * [0] + 18 * [2] + 22 * [0] + 58 * [3] + 50 * [0]
        )
    state = HeartKitState(
        dataId=random.randint(0, 100),
        app_state=random.choice(list(AppState)),
        data=data,
        segMask=seg_mask,
        results=HKResult(
            heart_rate=random.randint(20, 120),
            heart_rhythm=random.randint(0, 2),
            num_norm_beats=random.randint(0, 10),
            num_pac_beats=random.randint(0, 6),
            num_pvc_beats=random.randint(0, 3),
            arrhythmia=bool(random.randint(0, 2)),
        ),
    )
    set_global_state(state)


app = FastAPI(title="HeartKit", default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optionally allow serving front-end web app
static_app_path = os.getenv("WEB_APP_PATH", None)
if static_app_path:
    app.mount("/app", StaticFiles(directory=static_app_path, html=True), name="app")


@app.get("/")
def read_root():
    """Root"""
    # Serve front-end web app if defined
    if static_app_path:
        return RedirectResponse(url="/app")
    return {"ambiq": "ai"}


@app.get("/api/v1/state")
def get_state(state: HeartKitStateDep) -> HeartKitState:
    """Get State"""
    return state


@app.post("/api/v1/state")
def set_state(state: HeartKitState):
    """Set State"""
    set_global_state(state)


@app.get("/api/v1/app/state")
def get_app_state(state: HeartKitStateDep) -> AppState:
    """Get backend app state"""
    return state.app_state


@app.post("/api/v1/app/state")
def set_app_state(app_state: AppState, state: HeartKitStateDep):
    """Set backend app state"""
    state.app_state = app_state


@app.get("/api/v1/data_id")
def get_data_id(state: HeartKitStateDep, background_tasks: BackgroundTasks) -> int:
    """Get data id"""
    # background_tasks.add_task(emulate_data)
    return state.data_id


@app.post("/api/v1/data_id")
def set_data_id(data_id: int, state: HeartKitStateDep):
    """Set data id"""
    state.data_id = data_id


@app.get("/api/v1/data")
def get_data(state: HeartKitStateDep):
    """Get ECG data"""
    return state.data


@app.post("/api/v1/data")
def set_data(data: list[float], state: HeartKitStateDep):
    """Set ECG data"""
    state.data = data


@app.get("/api/v1/segmentation")
def get_segmentation(state: HeartKitStateDep) -> list[int]:
    """Get segmentation mask"""
    return state.seg_mask


@app.post("/api/v1/segmentation")
def set_segmentation(seg_mask: list[int], state: HeartKitStateDep):
    """Set segmentation mask"""
    state.seg_mask = seg_mask


@app.get("/api/v1/results")
def get_results(state: HeartKitStateDep) -> HKResult:
    """Get classification results"""
    return state.results


@app.post("/api/v1/results")
def set_results(results: HKResult, state: HeartKitStateDep):
    """Set classification results"""
    state.results = results


@app.on_event("startup")
def startup_event():
    """Server startup"""
    logger.info("REST Server starting up")
    # Start background worker


@app.on_event("shutdown")
def shutdown_event():
    """Server shhutdown"""
    logger.info("REST Server shutting down")


def run_forever():
    """Run server forever"""
    uvicorn.run(
        app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000"))
    )


if __name__ == "__main__":
    run_forever()
