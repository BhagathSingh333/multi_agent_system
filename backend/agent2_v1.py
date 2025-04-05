# ---------------------------------------------------------------------------
# Crop Planner – resilient version
# ---------------------------------------------------------------------------

import os, time
from textwrap import dedent
from typing import List, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.utils.log import logger

# Imports for WebScoket API
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool
import uuid
import json
from datetime import datetime

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found")

# ---------------------------------------------------------------------------
# Pydantic Models for Status Updates
# ---------------------------------------------------------------------------
class AgentStatusUpdate(BaseModel):
    event: str = "agent_status_update"
    agent: str  # Agent identifier key (e.g. "CropRecommenderBot")
    status: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class WorkflowMessage(BaseModel):
    event: str = "workflow_message"
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class WorkflowCompleted(BaseModel):
    event: str = "workflow_completed"
    result: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class CropsList(BaseModel):
    crops: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

class SoilRequirement(BaseModel):
    crop: str
    conditions: str

class SoilConditions(BaseModel):
    soil_requirements: List[SoilRequirement] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

class EnvironmentalData(BaseModel):
    crop: str
    climate_suitability: str
    rainfall: str
    temperature_min: float
    temperature_max: float
    temperature_optimal: float
    additional_info: str

class EnvironmentalResponse(BaseModel):
    environmental_factors: List[EnvironmentalData] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

class MarketingData(BaseModel):
    crop: str
    pricing_trends: str
    demand_forecast: str
    distribution_channels: str
    economic_viability: str

class MarketingResponse(BaseModel):
    marketing_factors: List[MarketingData] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

class SoilHealthData(BaseModel):
    crop: str
    soil_ph: float
    nutrient_levels: str
    water_retention: str
    organic_matter: str
    additional_info: str

class SoilHealthResponse(BaseModel):
    soil_health: List[SoilHealthData] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Agents  (unchanged prompts – only keyword structured_outputs)
# ---------------------------------------------------------------------------

def make_chat(max_tokens: int = 2500):
    return OpenAIChat(id="gpt-4o", api_key=openai_api_key, max_tokens=max_tokens)

crop_recommender_agent = Agent(
    model=make_chat(1500),
    tools=[DuckDuckGoTools()],
    description="You are **CropRecommenderBot**.",
    instructions=dedent(
        """\
        1. Search DuckDuckGo for agronomic guidance for the region.
        2. Return 5‑10 suitable crops with sources in `CropsList` format.
        """
    ),
    response_model=CropsList,
    structured_outputs=True,
)

soil_info_agent = Agent(
    model=make_chat(),
    tools=[DuckDuckGoTools()],
    description="You are **SoilInfoBot**.",
    instructions=dedent(
        """\
        For each crop supplied (comma‑separated) search for ideal soil conditions
        and return JSON that matches the `SoilConditions` schema.
        """
    ),
    response_model=SoilConditions,
    structured_outputs=True,
)

environmental_info_agent = Agent(
    model=make_chat(),
    tools=[DuckDuckGoTools()],
    description="You are **EnvironmentalInfoBot**.",
    instructions=dedent(
        """\
        Input: location + comma‑separated crop list …
        (full instructions unchanged)
        """
    ),
    response_model=EnvironmentalResponse,
    structured_outputs=True,
)

marketing_info_agent = Agent(
    model=make_chat(),
    tools=[DuckDuckGoTools()],
    description="You are **MarketingInfoBot**.",
    instructions=dedent(
        """\
        Input: location + comma‑separated crop list …
        (full instructions unchanged)
        """
    ),
    response_model=MarketingResponse,
    structured_outputs=True,
)

soil_health_info_agent = Agent(
    model=make_chat(),
    tools=[DuckDuckGoTools()],
    description="You are **SoilHealthInfoBot**.",
    instructions=dedent(
        """\
        Input: location + comma‑separated crop list …
        (full instructions unchanged)
        """
    ),
    response_model=SoilHealthResponse,
    structured_outputs=True,
)

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class CropPlannerWorkflow(Workflow):
    description = (
        "Recommend crops, then gather soil, environmental, marketing and "
        "soil‑health data for each crop."
    )
    status_callback: callable = lambda msg: None
    def safe_run(self, agent: Agent, prompt: str, empty):
        """Run an agent; return empty model on failure."""
        try:
            return agent.run(prompt).content
        except Exception as e:
            logger.warning(f"[CropPlanner] {agent.description.split()[1]} failed: {e}")
            return empty

    def run(self) -> RunResponse:
        location: str | None = getattr(self, "question", None)
        if not location:
            raise ValueError("Set `workflow.question` first")

        # 1. Crop Recommendation (synchronous)
        self.status_callback(
            AgentStatusUpdate(agent="CropRecommenderBot", status="starting crop recommendation").json()
        )
        crops_data: CropsList = self.safe_run(crop_recommender_agent, location, CropsList())
        crop_list = crops_data.crops
        # Update status: crop recommender completed and sent recommendations.
        self.status_callback(
            AgentStatusUpdate(
                agent="CropRecommenderBot", 
                status="completed crop recommendation; sent recommendations to SoilInfoBot, EnvironmentalInfoBot, and MarketingInfoBot"
            ).json()
        )

        # Prepare prompt for the next three agents.
        env_prompt = f"Location: {location}\nCrops: {', '.join(crop_list)}"

        # 2. Run Soil, Environmental, and Marketing agents in parallel.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Update each agent's status to "in progress" before submitting.
            future_soil = executor.submit(self.safe_run, soil_info_agent, ", ".join(crop_list), SoilConditions())
            future_env = executor.submit(self.safe_run, environmental_info_agent, env_prompt, EnvironmentalResponse())
            future_mkt = executor.submit(self.safe_run, marketing_info_agent, env_prompt, MarketingResponse())
            self.status_callback(
                AgentStatusUpdate(agent="SoilInfoBot", status="in progress: soil analysis").json()
            )
            self.status_callback(
                AgentStatusUpdate(agent="EnvironmentalInfoBot", status="in progress: environmental data collection").json()
            )
            self.status_callback(
                AgentStatusUpdate(agent="MarketingInfoBot", status="in progress: marketing data collection").json()
            )
            soil_data: SoilConditions = future_soil.result()
            env_data: EnvironmentalResponse = future_env.result()
            mkt_data: MarketingResponse = future_mkt.result()
        # Update statuses after parallel tasks complete.
        self.status_callback(
            AgentStatusUpdate(agent="SoilInfoBot", status="completed soil analysis").json()
        )
        self.status_callback(
            AgentStatusUpdate(agent="EnvironmentalInfoBot", status="completed environmental data collection").json()
        )
        self.status_callback(
            AgentStatusUpdate(agent="MarketingInfoBot", status="completed marketing data collection").json()
        )

        time.sleep(3)  # tiny pause to simulate delay or avoid rate limits

        # 3. Soil‑Health Factors (synchronous)
        self.status_callback(
            AgentStatusUpdate(agent="SoilHealthInfoBot", status="starting soil-health data collection").json()
        )
        sh_data: SoilHealthResponse = self.safe_run(soil_health_info_agent, env_prompt, SoilHealthResponse())
        self.status_callback(
            AgentStatusUpdate(agent="SoilHealthInfoBot", status="completed soil-health data collection").json()
        )

        # 4. Assemble Final Markdown
        md: List[str] = ["# Crop Planner Results", ""]
        for crop in crop_list:
            env = next((e for e in env_data.environmental_factors if e.crop == crop), None)
            mkt = next((m for m in mkt_data.marketing_factors if m.crop == crop), None)
            sh  = next((s for s in sh_data.soil_health if s.crop == crop), None)
            md.extend([
                f"## {crop}",
                "",
                f"**Soil requirements (LLM):** {next((i.conditions for i in soil_data.soil_requirements if i.crop == crop), 'N/A')}",
                "",
                "**Environmental factors:**",
                f"- Climate suitability: {getattr(env, 'climate_suitability', 'N/A')}",
                f"- Rainfall: {getattr(env, 'rainfall', 'N/A')}",
                f"- Temp min/max/optimal (°C): {getattr(env, 'temperature_min', '–')}/{getattr(env, 'temperature_max', '–')}/{getattr(env, 'temperature_optimal', '–')}",
                f"- Extra: {getattr(env, 'additional_info', 'N/A')}",
                "",
                "**Marketing factors:**",
                f"- Pricing trends: {getattr(mkt, 'pricing_trends', 'N/A')}",
                f"- Demand forecast: {getattr(mkt, 'demand_forecast', 'N/A')}",
                f"- Distribution channels: {getattr(mkt, 'distribution_channels', 'N/A')}",
                f"- Economic viability: {getattr(mkt, 'economic_viability', 'N/A')}",
                "",
                "**Soil‑health profile:**",
                f"- pH: {getattr(sh, 'soil_ph', 'N/A')}",
                f"- Nutrient levels: {getattr(sh, 'nutrient_levels', 'N/A')}",
                f"- Water retention: {getattr(sh, 'water_retention', 'N/A')}",
                f"- Organic matter: {getattr(sh, 'organic_matter', 'N/A')}",
                f"- Extra: {getattr(sh, 'additional_info', 'N/A')}",
                ""
            ])

        # 5. Sources (if any)
        if any([crops_data.sources, soil_data.sources, env_data.sources, mkt_data.sources, sh_data.sources]):
            md.extend(["---", "### Sources"])
            if crops_data.sources:
                md.append("**Crop recommendation**")
                md.extend(f"- {s}" for s in crops_data.sources)
            if soil_data.sources:
                md.append("**Soil requirements**")
                md.extend(f"- {s}" for s in soil_data.sources)
            if env_data.sources:
                md.append("**Environmental factors**")
                md.extend(f"- {s}" for s in env_data.sources)
            if mkt_data.sources:
                md.append("**Marketing factors**")
                md.extend(f"- {s}" for s in mkt_data.sources)
            if sh_data.sources:
                md.append("**Soil‑health factors**")
                md.extend(f"- {s}" for s in sh_data.sources)

        # 6. Send final workflow completion update.
        self.status_callback(WorkflowCompleted(result="\n".join(md)).json())
        return RunResponse("\n".join(md), RunEvent.workflow_completed)
    

app = FastAPI()

@app.websocket("/ws")
async def websocket_crop_planner(websocket: WebSocket):
    await websocket.accept()
    try:
        agent_statuses: Dict[str, dict] = {
            "CropRecommenderBot": {"id": str(uuid.uuid4()), "name": "Crop Recommender Bot", "status": "idle"},
            "SoilInfoBot": {"id": str(uuid.uuid4()), "name": "Soil Information Bot", "status": "idle"},
            "EnvironmentalInfoBot": {"id": str(uuid.uuid4()), "name": "Environmental Information Bot", "status": "idle"},
            "MarketingInfoBot": {"id": str(uuid.uuid4()), "name": "Marketing Information Bot", "status": "idle"},
            "SoilHealthInfoBot": {"id": str(uuid.uuid4()), "name": "Soil Health Information Bot", "status": "idle"}
        }
        initial_status = {"event": "initial_status", "agents": list(agent_statuses.values())}
        await websocket.send_text(json.dumps(initial_status))
        query = await websocket.receive_text()
        loop = asyncio.get_running_loop()
        def send_update(message: str):
            update_data = json.loads(message)
            if update_data.get("event") == "agent_status_update":
                agent_key = update_data.get("agent")
                if agent_key in agent_statuses:
                    # Update the stored status.
                    agent_statuses[agent_key]["status"] = update_data.get("status")
                    # Send the updated agent status.
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_text(json.dumps({
                            "event": "agent_status_update",
                            "agent": agent_statuses[agent_key]
                        })),
                        loop
                    )
            else:
                # For other events (workflow messages, completion), send directly.
                asyncio.run_coroutine_threadsafe(
                    websocket.send_text(message),
                    loop
                )
        print(f"Received query: {query}")
        await websocket.send_text("Received query. Starting crop recommendation...")
        wf = CropPlannerWorkflow(session_id="crop-planner-demo", storage=None)
        wf.question = query
        wf.status_callback = send_update
        await websocket.send_text("Running workflow...")
        
        result = await run_in_threadpool(wf.run)
        await websocket.send_text(result.content)
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Error: {e}")
        await websocket.close()

# ---------------------------------------------------------------------------
# CLI example
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     wf = CropPlannerWorkflow(session_id="crop-planner-demo", storage=None)
#     wf.question = "Punjab region of India"
#     result = wf.run()

#     print("\n=== Crop Planner Result ===\n")
#     print(result.content)
