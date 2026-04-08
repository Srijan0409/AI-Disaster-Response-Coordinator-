from openenv.core.env_server import create_app

from disaster_env.server.disaster_env_environment import DisasterEnvironment
from disaster_env.models import DisasterAction, DisasterObservation


# Create OpenEnv-compatible FastAPI app
app = create_app(
    DisasterEnvironment,   # pass class, not instance
    DisasterAction,
    DisasterObservation,
    max_concurrent_envs=4,
)


# ✅ Required for validator + local run
def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


# ✅ Required entrypoint
if __name__ == "__main__":
    main()