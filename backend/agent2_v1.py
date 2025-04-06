# ---------------------------------------------------------------------------
# Crop Planner – resilient + verification loop + summarization + WebSockets
# ---------------------------------------------------------------------------
# pip install agno openai python‑dotenv duckduckgo‑search pydantic fastapi uvicorn
# ---------------------------------------------------------------------------

import os, time, json, logging
from textwrap import dedent
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.utils.log import logger

# Imports for WebSocket API
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool
import uuid

# ---------------------------------------------------------------------------
# Base Model for Agent Outputs
# ---------------------------------------------------------------------------
class AgentOutput(BaseModel):
    thinking: str = None

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found")

# ---------------------------------------------------------------------------
# Pydantic Models for Status Updates and Log Updates
# ---------------------------------------------------------------------------
class AgentStatusUpdate(BaseModel):
    event: str = "agent_status_update"
    agent: str  # Agent identifier key (e.g. "CropRecommenderAgent")
    status: str
    log: str = None
    next_agent: List[str] = Field(default_factory=list)  # List of next agents to be called
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class WorkflowMessage(BaseModel):
    event: str = "workflow_message"
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class WorkflowCompleted(BaseModel):
    event: str = "workflow_completed"
    result: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class LogUpdate(BaseModel):
    event: str = "log_update"
    log: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class CropsList(AgentOutput):
    crops: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

class EnvironmentalData(BaseModel):
    crop: str
    climate_suitability: str
    rainfall: str
    temperature_min: float
    temperature_max: float
    temperature_optimal: float
    additional_info: str

class EnvironmentalResponse(AgentOutput):
    environmental_factors: List[EnvironmentalData] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

class MarketingData(BaseModel):
    crop: str
    pricing_trends: str
    demand_forecast: str
    distribution_channels: str
    economic_viability: str

class MarketingResponse(AgentOutput):
    marketing_factors: List[MarketingData] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

class SoilHealthData(BaseModel):
    crop: str
    soil_ph: float
    nutrient_levels: str
    water_retention: str
    organic_matter: str
    additional_info: str

class SoilHealthResponse(AgentOutput):
    soil_health: List[SoilHealthData] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

# ---- Verification Schemas ------------------------------------------------

class VerificationItem(BaseModel):
    crop: str
    missing_domains: List[str]  # e.g. ["environmental", "marketing", "soil_health"]

class VerificationResponse(AgentOutput):
    missing_info: List[VerificationItem] = Field(default_factory=list)

# ---- Summarization Schema -------------------------------------------------

class SummaryReport(AgentOutput):
    report: str = Field(..., description="A detailed summary report including recommendations and rationale with clean markdown format.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chat(max_tokens: int = 3000):
    return OpenAIChat(id="gpt-4o", api_key=openai_api_key, max_tokens=max_tokens)

def thinking_prompt(original: str) -> str:
    """Prepends the 'thinking' requirement to an instruction string."""
    return dedent(
        "1. For the following tasks, store your reasoning in a variable named "
        "`thinking`. Return `thinking` at the beginning of your response as JSON format.\n"
    ) + original

# ---------------------------------------------------------------------------
# Agents  (detailed instructions remain unchanged)
# ---------------------------------------------------------------------------

crop_recommender_agent = Agent(
    model=make_chat(1500),
    tools=[DuckDuckGoTools()],
    description="You are **CropRecommenderAgent**.",
    instructions=thinking_prompt(
        """2. Search DuckDuckGo for agronomic guidance for the region.
           3. Return 5‑10 suitable crops with sources in `CropsList` format."""
    ),
    response_model=CropsList,
    structured_outputs=True,
)

environmental_info_agent = Agent(
    model=make_chat(),
    tools=[DuckDuckGoTools()],
    description="You are **EnvironmentalInfoAgent**.",
    instructions=thinking_prompt(
        """Input: a location followed by a comma‑separated crop list.
           For each crop provide:
             • climate_suitability
             • rainfall
             • temperature_min / _max / _optimal (°C)
             • additional_info
           Return JSON matching the `EnvironmentalResponse` schema with sources."""
    ),
    response_model=EnvironmentalResponse,
    structured_outputs=True,
)

marketing_info_agent = Agent(
    model=make_chat(),
    tools=[DuckDuckGoTools()],
    description="You are **MarketingInfoAgent**.",
    instructions=thinking_prompt(
        """Input: location + crop list.
           For each crop summarise:
             • pricing_trends   • demand_forecast
             • distribution_channels   • economic_viability
           Return JSON in `MarketingResponse` format with sources."""
    ),
    response_model=MarketingResponse,
    structured_outputs=True,
)

soil_health_info_agent = Agent(
    model=make_chat(),
    tools=[DuckDuckGoTools()],
    description="You are **SoilHealthInfoAgent**.",
    instructions=thinking_prompt(
        """Input: location + crop list.
           For each crop return soil_ph, nutrient_levels, water_retention,
           organic_matter and additional_info in `SoilHealthResponse` format."""
    ),
    response_model=SoilHealthResponse,
    structured_outputs=True,
)

verification_agent = Agent(
    model=make_chat(1200),
    description="You are **VerificationAgent**.",
    instructions=thinking_prompt(
        """You receive a JSON object for ONE crop with three keys:
           `environmental`, `marketing`, `soil_health`.
           Any field that is empty, missing or has the literal string "N/A"
           means that domain is incomplete for this crop.
           Return JSON that matches the `VerificationResponse` schema, e.g.:
           {
             "missing_info": [
               { "crop": "Wheat",
                 "missing_domains": ["marketing"] }
             ]
           }"""
    ),
    response_model=VerificationResponse,
    structured_outputs=True,
)

# --- New summarization agent -----------------------------------------------

summary_agent = Agent(
    model=make_chat(1800),
    tools=[DuckDuckGoTools()],
    description="You are **SummaryAgent**.",
    instructions=thinking_prompt(
        """Input: You receive the final structured data for a location along with the aggregated information for each crop.
           Your task is to produce a detailed summary report that:
           1. Reviews the environmental, marketing, and soil‑health information for each crop.
           2. Identifies the best crops to grow in the region along with valid reasons for each choice.
           3. Provides actionable suggestions to farmers about crop selection and cultivation practices based on the data.
           4. The report should be well-structured and formatted for easy reading.
           5. Your response should be a detailed report in Markdown format."""
    ),
    response_model=SummaryReport,
    structured_outputs=True,
)

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class CropPlannerWorkflow(Workflow):
    description = (
        "Recommend crops, gather data from three domain agents, run a "
        "verification pass, and then repeatedly refill missing data until "
        "all fields are complete (or until 3 retries). If some information is still "
        "missing, fill it with a default message stating that the information may not be open source "
        "and may be behind paid sources. Finally, summarize the results with recommendations for farmers."
    )
    FALLBACK_MESSAGE = ("The information may not be open source and may be behind paid sources. ")
    status_callback: callable = lambda msg: None

    def safe_run(self, agent: Agent, prompt: str, empty):
        """Run an agent; return its .content or a default empty model."""
        try:
            res = agent.run(prompt)
            # Send status update via callback if defined
            agent_name = agent.description.split()[2].replace("*", "")[:-1]
            
            self.log_steps.append({
                "agent": agent_name,
                "prompt": prompt,
                "response": json.dumps(
                    res.content.dict() if hasattr(res.content, 'dict') else res.content, 
                    indent=2
                )
            })

            return res.content
        except Exception as e:
            agent_name = agent.description.split()[2].replace("*", "")[:-1]
            logger.warning(f"[CropPlanner] {agent_name} failed: {e}")
            self.log_steps.append({
                "agent":agent_name, 
                "ERROR":e
                })
            
            # Send error status via callback if defined
            self.status_callback(
                AgentStatusUpdate(
                    agent=agent_name, 
                    status=f"error: {str(e)[:100]}"
                ).json()
            )
                
            return empty

    def run(self) -> RunResponse:
        self.log_steps: List[Dict[str, Any]] = []  # initialize conversation log

        location: str | None = getattr(self, "question", None)
        if not location:
            raise ValueError("Set `workflow.question` first")

        # -- 1. crop recommendation ---------------------------------------
        
        self.status_callback(
            AgentStatusUpdate(
                agent="CropRecommenderAgent", 
                status="Started crop recommendation"
            ).json()
        )
            
        crops_data: CropsList = self.safe_run(
            crop_recommender_agent, location, CropsList()
        )
        crop_list = crops_data.crops
        self.status_callback(
            AgentStatusUpdate(
                agent="CropRecommenderAgent", 
                status="Completed task",
                log=self.log_steps[-1]['response'],
                next_agent=["SoilHealthInfoAgent", "EnvironmentalInfoAgent", "MarketingInfoAgent"]
            ).json()
        )

        # -- 2. initial domain calls --------------------------------------
        env_prompt = f"Location: {location}\nCrops: {', '.join(crop_list)}"
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            future_soil = executor.submit(self.safe_run, soil_health_info_agent, env_prompt, SoilHealthResponse())
            future_env = executor.submit(self.safe_run, environmental_info_agent, env_prompt, EnvironmentalResponse())
            future_mkt = executor.submit(self.safe_run, marketing_info_agent, env_prompt, MarketingResponse())
            self.status_callback(
                AgentStatusUpdate(agent="SoilHealthInfoAgent", status="Started: soil analysis").json()
            )
            self.status_callback(
                AgentStatusUpdate(agent="EnvironmentalInfoAgent", status="Started: environmental data collection").json()
            )
            self.status_callback(
                AgentStatusUpdate(agent="MarketingInfoAgent", status="Started: marketing data collection").json()
            )
            soil_data: SoilHealthResponse = future_soil.result()
            env_data: EnvironmentalResponse = future_env.result()
            mkt_data: MarketingResponse = future_mkt.result()
            self.status_callback(
                AgentStatusUpdate(
                    agent="SoilHealthInfoAgent", 
                    status="Completed soil analysis sending info to VerificationAgent",
                    log=self.log_steps[-1]['response'],
                    next_agent=["VerificationAgent"]
                    ).json()
            )
            self.status_callback(
                AgentStatusUpdate(
                    agent="EnvironmentalInfoAgent", 
                    status="Completed environmental data collection sending info to VerificationAgent",
                    log=self.log_steps[-1]['response'],
                    next_agent=["VerificationAgent"]
                    ).json()
            )
            self.status_callback(
                AgentStatusUpdate(
                    agent="MarketingInfoAgent", 
                    status="Completed marketing data collection sending info to VerificationAgent",
                    log=self.log_steps[-1]['response'],
                    next_agent=["VerificationAgent"]
                    ).json()
            )

        time.sleep(3)
        sh_map = {d.crop: d for d in soil_data.soil_health}
        env_map = {d.crop: d for d in env_data.environmental_factors}
        mkt_map = {d.crop: d for d in mkt_data.marketing_factors}


        max_retries = 3
        retry = 0
        stopVerification = False

        self.status_callback(
            AgentStatusUpdate(
                agent="VerificationAgent", 
                status="Started verification loop"
            ).json()
        )
        while retry < max_retries:
            # For each retry, reinitialize missing lists
            missing_env = []
            missing_mkt = []
            missing_soil = []

            # Verify each crop's data using the verifier agent
            for crop in crop_list:
                combined = {
                    "crop": crop,
                    "environmental": env_map.get(crop, {}).dict() if crop in env_map else {},
                    "marketing": mkt_map.get(crop, {}).dict() if crop in mkt_map else {},
                    "soil_health": sh_map.get(crop, {}).dict() if crop in sh_map else {},
                }
                ver_prompt = json.dumps(combined, indent=2)
                ver_resp: VerificationResponse = self.safe_run(
                    verification_agent, ver_prompt, VerificationResponse()
                )
                
                # If the verifier returns missing domains, add the crop to the appropriate lists
                if ver_resp.missing_info and ver_resp.missing_info[0].missing_domains:
                    missing = ver_resp.missing_info[0].missing_domains
                    if "environmental" in missing:
                        missing_env.append(crop)
                    if "marketing" in missing:
                        missing_mkt.append(crop)
                    if "soil_health" in missing:
                        missing_soil.append(crop)
            
            logger.info(f"Retry {retry}: missing_env: {missing_env}, missing_mkt: {missing_mkt}, missing_soil: {missing_soil}")
            
            # If no missing data remains, break out of the loop early
            if not missing_env and not missing_mkt and not missing_soil:
                self.status_callback(
                    AgentStatusUpdate(
                        agent="VerificationAgent",
                        status="Completed task",
                        log=self.log_steps[-1]['response'],
                        next_agent=["SummaryAgent"]
                    ).json()
                )
                stopVerification = True
                logger.info(f"All data verified successfully for all crops.")
                break

            # Refill missing data for all domains in parallel
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                future_env = None
                future_mkt = None
                future_soil = None

                if missing_env:
                    env_prompt_missing = f"Location: {location}\nCrops: {', '.join(missing_env)}"
                    future_env = executor.submit(
                        self.safe_run, environmental_info_agent, env_prompt_missing, EnvironmentalResponse()
                    )
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="EnvironmentalInfoAgent", 
                            status="Started: environmental data collection"
                        ).json()
                    )
                    logger.info(f"env_prompt_missing: {env_prompt_missing} {datetime.utcnow().isoformat()}")

                if missing_mkt:
                    mkt_prompt_missing = f"Location: {location}\nCrops: {', '.join(missing_mkt)}"
                    future_mkt = executor.submit(
                        self.safe_run, marketing_info_agent, mkt_prompt_missing, MarketingResponse()
                    )
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="MarketingInfoAgent", 
                            status="Started: marketing data collection"
                        ).json()
                    )
                    logger.info(f"mkt_prompt_missing: {mkt_prompt_missing} {datetime.utcnow().isoformat()}")

                if missing_soil:
                    soil_prompt_missing = f"Location: {location}\nCrops: {', '.join(missing_soil)}"
                    future_soil = executor.submit(
                        self.safe_run, soil_health_info_agent, soil_prompt_missing, SoilHealthResponse()
                    )
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="SoilHealthInfoAgent", 
                            status="Started: soil analysis"
                        ).json()
                    )
                    logger.info(f"soil_prompt_missing: {soil_prompt_missing} {datetime.utcnow().isoformat()}")

                # Update maps with the results from parallel calls
                if future_env:
                    env_refill: EnvironmentalResponse = future_env.result()
                    for env_data in env_refill.environmental_factors:
                        env_map[env_data.crop] = env_data
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="EnvironmentalInfoAgent", 
                            status="Completed task",
                            log=self.log_steps[-1]['response'],
                            next_agent=["VerificationAgent"]
                        ).json()
                    )

                if future_mkt:
                    mkt_refill: MarketingResponse = future_mkt.result()
                    for mkt_data in mkt_refill.marketing_factors:
                        mkt_map[mkt_data.crop] = mkt_data
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="MarketingInfoAgent", 
                            status="Completed task",
                            log=self.log_steps[-1]['response'],
                            next_agent=["VerificationAgent"]
                        ).json()
                    )

                if future_soil:
                    soil_refill: SoilHealthResponse = future_soil.result()
                    for sh_data in soil_refill.soil_health:
                        sh_map[sh_data.crop] = sh_data
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="SoilHealthInfoAgent", 
                            status="Completed task",
                            log=self.log_steps[-1]['response'],
                            next_agent=["VerificationAgent"]
                        ).json()
                    )

            retry += 1
            time.sleep(1)  # Optional delay between retries

        # After max retries, for any crop still missing data, set fallback values.
        for crop in crop_list:
            # Re-run verification to check for missing domains one last time.
            combined = {
                "crop": crop,
                "environmental": env_map.get(crop, {}).dict() if crop in env_map else {},
                "marketing": mkt_map.get(crop, {}).dict() if crop in mkt_map else {},
                "soil_health": sh_map.get(crop, {}).dict() if crop in sh_map else {},
            }
            ver_prompt = json.dumps(combined, indent=2)
            ver_resp: VerificationResponse = self.safe_run(
                verification_agent, ver_prompt, VerificationResponse()
            )
            if ver_resp.missing_info and ver_resp.missing_info[0].missing_domains:
                missing = ver_resp.missing_info[0].missing_domains
                if "environmental" in missing and crop not in env_map:
                    env_map[crop] = EnvironmentalData(
                        crop=crop,
                        climate_suitability=self.FALLBACK_MESSAGE,
                        rainfall=self.FALLBACK_MESSAGE,
                        temperature_min=0,
                        temperature_max=0,
                        temperature_optimal=0,
                        additional_info=self.FALLBACK_MESSAGE
                    )
                    logger.info(f"Fallback for environmental data for {crop}")
                if "marketing" in missing and crop not in mkt_map:
                    mkt_map[crop] = MarketingData(
                        crop=crop,
                        pricing_trends=self.FALLBACK_MESSAGE,
                        demand_forecast=self.FALLBACK_MESSAGE,
                        distribution_channels=self.FALLBACK_MESSAGE,
                        economic_viability=self.FALLBACK_MESSAGE
                    )
                    logger.info(f"Fallback for marketing data for {crop}")
                if "soil_health" in missing and crop not in sh_map:
                    sh_map[crop] = SoilHealthData(
                        crop=crop,
                        soil_ph=0,
                        nutrient_levels=self.FALLBACK_MESSAGE,
                        water_retention=self.FALLBACK_MESSAGE,
                        organic_matter=self.FALLBACK_MESSAGE,
                        additional_info=self.FALLBACK_MESSAGE
                    )
                    logger.info(f"Fallback for soil health data for {crop}")
        if stopVerification == False:
            logger.info(f"Final verification loop completed with missing data: {missing_env}, {missing_mkt}, {missing_soil}")
            self.status_callback(
                AgentStatusUpdate(
                    agent="VerificationAgent", 
                    status="Completed task",
                    log=self.log_steps[-1]['response'],
                    next_agent=["SummaryAgent"]
                ).json()
            )

        # -- 4. Build per-crop report ---------------------------------------
        md: List[str] = ["# Crop Planner Detailed Report", ""]
        final_structured: Dict[str, Dict] = {}
        for crop in crop_list:
            env = env_map.get(crop)
            mkt = mkt_map.get(crop)
            sh  = sh_map.get(crop)
            md.extend(
                [
                    f"## {crop}",
                    "",
                    "**Environmental factors:**",
                    f"- Climate suitability: {getattr(env, 'climate_suitability', 'N/A')}",
                    f"- Rainfall: {getattr(env, 'rainfall', 'N/A')}",
                    f"- Temp min/max/optimal (°C): "
                    f"{getattr(env, 'temperature_min', '–')}/"
                    f"{getattr(env, 'temperature_max', '–')}/"
                    f"{getattr(env, 'temperature_optimal', '–')}",
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
                    "",
                ]
            )
            final_structured[crop] = {
                "environmental": env.dict() if env else {},
                "marketing": mkt.dict() if mkt else {},
                "soil_health": sh.dict() if sh else {},
            }
       
        # -- 5. Summarize the complete information using SummaryAgent --------
        self.status_callback(
                AgentStatusUpdate(
                    agent="VerificationAgent", 
                    status="Completed task",
                    log=self.log_steps[-1]['response'],
                    next_agent=["SummaryAgent"]
                ).json()
                )
        
        self.status_callback(
            AgentStatusUpdate(
                agent="SummaryAgent", 
                status="Started summarization"
            ).json()
        )
            
        summary_prompt = dedent(f"""
            Location: {location}
            Final structured data:
            {json.dumps(final_structured, indent=2)}
            
            Based on the above data, produce a detailed summary report that:
            - Reviews the environmental, marketing, and soil‑health conditions for the region.
            - Identifies the best crops to grow in {location} and provides reasons for each choice.
            - Offers actionable suggestions to farmers for improving yields and sustainability.
            Your response should be a detailed report.
        """)
        summary_report: SummaryReport = self.safe_run(
            summary_agent, summary_prompt, SummaryReport(report="N/A")
        )

        # -- 6. Append the summary report to the final output ---------------
        md.extend(
            [
                "---",
                "## Summary Report",
                "",
                summary_report.report,
                "",
            ]
        )

        # -- 7. Append conversation log -------------------------------------

        # md.extend(["## Agent Conversation Log", ""])
        # md.extend([json.dumps(log, indent=2) if isinstance(log, dict) else log for log in self.log_steps])

        self.status_callback(
            AgentStatusUpdate(
                agent="SummaryAgent", 
                status="Completed task",
                log=self.log_steps[-1]['response']
            ).json()
        )
        
        self.status_callback(
            WorkflowCompleted(
                result=summary_report.report
            ).json()
        )

        return RunResponse(summary_report.report, RunEvent.workflow_completed)

# ---------------------------------------------------------------------------
# FastAPI App & WebSocket Endpoint
# ---------------------------------------------------------------------------
app = FastAPI()

@app.websocket("/ws")
async def websocket_crop_planner(websocket: WebSocket):
    await websocket.accept()
   
    # Set up agent statuses once
    agent_statuses: Dict[str, dict] = {
        "CropRecommenderAgent": {"id": str(uuid.uuid4()), "name": "Crop Recommender Agent", "status": "idle"},
        "EnvironmentalInfoAgent": {"id": str(uuid.uuid4()), "name": "Environmental Information Agent", "status": "idle"},
        "MarketingInfoAgent": {"id": str(uuid.uuid4()), "name": "Marketing Information Agent", "status": "idle"},
        "SoilHealthInfoAgent": {"id": str(uuid.uuid4()), "name": "Soil Information Agent", "status": "idle"},
        "VerificationAgent": {"id": str(uuid.uuid4()), "name": "Verification Agent", "status": "idle"},
        "SummaryAgent": {"id": str(uuid.uuid4()), "name": "Summary Agent", "status": "idle"},
    }
   
    # Get current event loop
    loop = asyncio.get_running_loop()
   
    # Create log handler once
    # async def send_log(msg: str):
    #     await websocket.send_text(msg)
       
    # streaming_handler = StreamingLogHandler(send_log, loop)
    # streaming_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # logger.addHandler(streaming_handler)
   
    try:
        # Send initial status to client
        initial_status = {"event": "initial_status", "agents": list(agent_statuses.values())}
        await websocket.send_text(json.dumps(initial_status))
       
        # Continuous loop to handle multiple queries
        while True:
            # Reset agent statuses to idle between queries
            for key in agent_statuses:
                agent_statuses[key]["status"] = "idle"
           
            # Send updated status
            await websocket.send_text(json.dumps({
                "event": "agents_reset",
                "agents": list(agent_statuses.values())
            }))
           
            # Wait for the next query
            query = await websocket.receive_text()
            print(f"Received query: {query}")
           
            # Skip empty queries
            if not query or query.strip() == "":
                await websocket.send_text(json.dumps({
                    "event": "error",
                    "message": "Empty query received. Please provide a location."
                }))
                continue
               
            # Define callback for this query
            def send_update(message: str):
                update_data = json.loads(message)
                if update_data.get("event") == "agent_status_update":
                    agent_key = update_data.get("agent")
                    if agent_key in agent_statuses:
                        agent_statuses[agent_key]["status"] = update_data.get("status")
                        agent_statuses[agent_key]["log"] = update_data.get("log")
                        agent_statuses[agent_key]["next_agent"] = update_data.get("next_agent")
                        agent_statuses[agent_key]["timestamp"] = update_data.get("timestamp")
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text(json.dumps({
                                "event": "agent_status_update",
                                "agent": agent_statuses[agent_key]
                            })),
                            loop
                        )
                elif update_data.get("event") == "workflow_completed":
                    # When workflow is completed, send the final result
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_text(json.dumps({
                            "event": "workflow_completed",
                            "result": update_data.get("result")
                        })),
                        loop
                    )
                else:
                    # Forward other messages directly
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_text(message),
                        loop
                    )
           
            # Notify client that workflow is starting
            wf_msg = WorkflowMessage(message=f"Running Crop Planner workflow for: {query}").json()
            await websocket.send_text(wf_msg)
            # Initialize and run the workflow
            wf = CropPlannerWorkflow(session_id=f"crop-planner-{uuid.uuid4()}", storage=None)
            wf.question = query
            wf.status_callback = send_update
            # Run the workflow in a separate thread to avoid blocking the event loop
            await run_in_threadpool(wf.run)
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        try:
            await websocket.send_text(json.dumps({
                "event": "error",
                "message": f"An error occurred: {str(e)}"
            }))
        except:
            pass
# ---------------------------------------------------------------------------
# CLI example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    
    # Check if running as script with CLI argument
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run in CLI mode
        wf = CropPlannerWorkflow(session_id="crop-planner-demo", storage=None)
        wf.question = "what crops are better in Chicago USA?"
        result = wf.run()
        print("\n=== Crop Planner Result ===\n")
        print(result.content)
    else:
        # Run as API server
        print("Starting Crop Planner WebSocket server...")
        print("Connect to ws://localhost:8000/ws")
        uvicorn.run(app, host="0.0.0.0", port=8000)
