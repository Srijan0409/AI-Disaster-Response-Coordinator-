# =============================================================================
# app.py - FastAPI server for AI Disaster Response Coordinator
# =============================================================================

from openenv.core.env_server import create_app

from disaster_env.server.disaster_env_environment import DisasterEnvironment
from disaster_env.models import DisasterAction, DisasterObservation

app = create_app(
    DisasterEnvironment,
    DisasterAction,
    DisasterObservation,
    max_concurrent_envs=8,
)

# Health check - required by the OpenEnv validator and HF Space ping check
@app.get("/")
def health():
    return {"status": "ok", "env": "disaster_env"}

app.title       = "AI Disaster Response Environment"
app.description = "OpenEnv-compatible disaster simulation for rescue optimisation"


def main(host: str = "0.0.0.0", port: int = 8000):
    # uvicorn imported here so loading this module doesn't require uvicorn
    # to be installed - e.g. when the OpenEnv harness imports the app object
    # without intending to serve it directly.
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()