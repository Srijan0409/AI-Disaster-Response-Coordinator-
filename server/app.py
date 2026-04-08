from openenv.core.env_server import create_app
from fastapi import FastAPI

from disaster_env.server.disaster_env_environment import DisasterEnvironment
from disaster_env.models import DisasterAction, DisasterObservation

# Create OpenEnv-compatible FastAPI app

app = create_app(
DisasterEnvironment,
DisasterAction,
DisasterObservation,
max_concurrent_envs=8,   # increased for better performance
)

# ✅ Health check (VERY IMPORTANT)

@app.get("/")
def health():
return {"status": "ok", "env": "disaster_env"}

# ✅ Metadata (optional but good)

app.title = "AI Disaster Response Environment"
app.description = "OpenEnv-compatible disaster simulation for rescue optimization"

# ✅ Required for validator + local run

def main(host: str = "0.0.0.0", port: int = 8000):
import uvicorn
uvicorn.run(app, host=host, port=port)

# ✅ Entry point

if **name** == "**main**":
main()
