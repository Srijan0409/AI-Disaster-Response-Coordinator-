from openenv.core.env_server import create_app

from disaster_env.server.disaster_env_environment import DisasterEnvironment
from disaster_env.models import DisasterAction, DisasterObservation

# Create OpenEnv-compatible FastAPI app
app = create_app(
    DisasterEnvironment,   # IMPORTANT: pass class, not instance
    DisasterAction,
    DisasterObservation,
    max_concurrent_envs=4,  # optional but recommended
)