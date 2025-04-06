import os, time, json, logging
from textwrap import dedent
from typing import List, Dict, Any, Optional
from datetime import datetime
import concurrent.futures
import threading

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.utils.log import logger

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool
import uuid


class AgentOutput(BaseModel):
    thinking: Optional[str] = None


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found")


class AgentStatusUpdate(BaseModel):
    event: str = "agent_status_update"
    agent: str
    status: str
    log: Optional[str] = None
    next_agent: List[str] = Field(default_factory=list)
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


class VerificationItem(BaseModel):
    crop: str
    missing_domains: List[str]


class VerificationResponse(AgentOutput):
    missing_info: List[VerificationItem] = Field(default_factory=list)


class SummaryReport(AgentOutput):
    report: str = Field(
        ...,
        description="A detailed summary report including recommendations and rationale with clean markdown format.",
    )


def make_chat(max_tokens: int = 2000):
    return OpenAIChat(id="gpt-4o", api_key=openai_api_key, max_tokens=max_tokens)


def thinking_prompt(original: str) -> str:
    return (
        dedent(
            "1. For the following tasks, store your reasoning in a variable named "
            "`thinking`. Return `thinking` at the beginning of your response as JSON format.\n"
        )
        + original
    )


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


class CropPlannerWorkflow(Workflow):
    description = (
        "Recommend crops, gather data from three domain agents, verify information, "
        "and summarize results with recommendations for farmers."
    )
    FALLBACK_MESSAGE = "Information may be behind paid sources. "
    status_callback: callable = lambda msg: None

    def safe_run(self, agent: Agent, prompt: str, empty):
        agent_name = agent.description.split()[2].replace("*", "")[:-1]

        try:
            # self.status_callback(
            #     AgentStatusUpdate(
            #         agent=agent_name,
            #         status="Processing..."
            #     ).json()
            # )

            result = {"completed": False, "response": None, "error": None}

            def run_with_timeout():
                try:
                    result["response"] = agent.run(prompt)
                    result["completed"] = True
                except Exception as e:
                    result["error"] = e

            thread = threading.Thread(target=run_with_timeout)
            thread.daemon = True
            thread.start()

            thread.join(120)

            if not result["completed"]:
                logger.warning(
                    f"[CropPlanner] {agent_name} timed out after 120 seconds"
                )
                self.status_callback(
                    AgentStatusUpdate(
                        agent=agent_name, status="Timed out after 120 seconds"
                    ).json()
                )
                return empty

            if result["error"]:
                raise result["error"]

            res = result["response"]

            self.log_steps.append(
                {
                    "agent": agent_name,
                    "prompt": prompt,
                    "response": json.dumps(
                        (
                            res.content.dict()
                            if hasattr(res.content, "dict")
                            else res.content
                        ),
                        indent=2,
                    ),
                }
            )

            return res.content
        except Exception as e:
            logger.warning(f"[CropPlanner] {agent_name} failed: {e}")
            self.log_steps.append({"agent": agent_name, "ERROR": str(e)})

            self.status_callback(
                AgentStatusUpdate(
                    agent=agent_name, status=f"error: {str(e)[:100]}"
                ).json()
            )

            return empty

    def run(self) -> RunResponse:
        self.log_steps: List[Dict[str, Any]] = []

        location: str | None = getattr(self, "question", None)
        if not location:
            raise ValueError("Set `workflow.question` first")

        self.status_callback(
            AgentStatusUpdate(
                agent="CropRecommenderAgent", status="Started crop recommendation"
            ).json()
        )

        crops_data: CropsList = self.safe_run(
            crop_recommender_agent, location, CropsList()
        )
        crop_list = crops_data.crops

        if len(crop_list) > 5:
            logger.info(
                f"Limiting crops from {len(crop_list)} to 5 for better performance"
            )
            crop_list = crop_list[:5]

        self.status_callback(
            AgentStatusUpdate(
                agent="CropRecommenderAgent",
                status="Completed task",
                log=self.log_steps[-1]["response"],
                next_agent=[
                    "SoilHealthInfoAgent",
                    "EnvironmentalInfoAgent",
                    "MarketingInfoAgent",
                ],
            ).json()
        )

        env_prompt = f"Location: {location}\nCrops: {', '.join(crop_list)}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_soil = executor.submit(
                self.safe_run, soil_health_info_agent, env_prompt, SoilHealthResponse()
            )
            future_env = executor.submit(
                self.safe_run,
                environmental_info_agent,
                env_prompt,
                EnvironmentalResponse(),
            )
            future_mkt = executor.submit(
                self.safe_run, marketing_info_agent, env_prompt, MarketingResponse()
            )
            self.status_callback(
                AgentStatusUpdate(
                    agent="SoilHealthInfoAgent", status="Started: soil analysis"
                ).json()
            )
            self.status_callback(
                AgentStatusUpdate(
                    agent="EnvironmentalInfoAgent",
                    status="Started: environmental data collection",
                ).json()
            )
            self.status_callback(
                AgentStatusUpdate(
                    agent="MarketingInfoAgent",
                    status="Started: marketing data collection",
                ).json()
            )
            soil_data: SoilHealthResponse = future_soil.result()
            env_data: EnvironmentalResponse = future_env.result()
            mkt_data: MarketingResponse = future_mkt.result()
            self.status_callback(
                AgentStatusUpdate(
                    agent="SoilHealthInfoAgent",
                    status="Completed soil analysis sending info to VerificationAgent",
                    log=self.log_steps[-1]["response"],
                    next_agent=["VerificationAgent"],
                ).json()
            )
            self.status_callback(
                AgentStatusUpdate(
                    agent="EnvironmentalInfoAgent",
                    status="Completed environmental data collection sending info to VerificationAgent",
                    log=self.log_steps[-1]["response"],
                    next_agent=["VerificationAgent"],
                ).json()
            )
            self.status_callback(
                AgentStatusUpdate(
                    agent="MarketingInfoAgent",
                    status="Completed marketing data collection sending info to VerificationAgent",
                    log=self.log_steps[-1]["response"],
                    next_agent=["VerificationAgent"],
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
                agent="VerificationAgent", status="Started verification loop"
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
                    "environmental": (
                        env_map.get(crop, {}).dict() if crop in env_map else {}
                    ),
                    "marketing": (
                        mkt_map.get(crop, {}).dict() if crop in mkt_map else {}
                    ),
                    "soil_health": (
                        sh_map.get(crop, {}).dict() if crop in sh_map else {}
                    ),
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

            logger.info(
                f"Retry {retry}: missing_env: {missing_env}, missing_mkt: {missing_mkt}, missing_soil: {missing_soil}"
            )

            # If no missing data remains, break out of the loop early
            if not missing_env and not missing_mkt and not missing_soil:
                self.status_callback(
                    AgentStatusUpdate(
                        agent="VerificationAgent",
                        status="Completed task",
                        log=self.log_steps[-1]["response"],
                        next_agent=["SummaryAgent"],
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
                    env_prompt_missing = (
                        f"Location: {location}\nCrops: {', '.join(missing_env)}"
                    )
                    future_env = executor.submit(
                        self.safe_run,
                        environmental_info_agent,
                        env_prompt_missing,
                        EnvironmentalResponse(),
                    )
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="VerificationAgent",
                            status="Started",
                            log=self.log_steps[-1]["response"],
                            next_agent=["EnvironmentalInfoAgent"],
                        ).json()
                    )
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="EnvironmentalInfoAgent",
                            status="Started: environmental data collection",
                        ).json()
                    )
                    logger.info(
                        f"env_prompt_missing: {env_prompt_missing} {datetime.utcnow().isoformat()}"
                    )

                if missing_mkt:
                    mkt_prompt_missing = (
                        f"Location: {location}\nCrops: {', '.join(missing_mkt)}"
                    )
                    future_mkt = executor.submit(
                        self.safe_run,
                        marketing_info_agent,
                        mkt_prompt_missing,
                        MarketingResponse(),
                    )
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="VerificationAgent",
                            status="Started",
                            log=self.log_steps[-1]["response"],
                            next_agent=["MarketingInfoAgent"],
                        ).json()
                    )
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="MarketingInfoAgent",
                            status="Started: marketing data collection",
                        ).json()
                    )
                    logger.info(
                        f"mkt_prompt_missing: {mkt_prompt_missing} {datetime.utcnow().isoformat()}"
                    )

                if missing_soil:
                    soil_prompt_missing = (
                        f"Location: {location}\nCrops: {', '.join(missing_soil)}"
                    )
                    future_soil = executor.submit(
                        self.safe_run,
                        soil_health_info_agent,
                        soil_prompt_missing,
                        SoilHealthResponse(),
                    )
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="VerificationAgent",
                            status="Started",
                            log=self.log_steps[-1]["response"],
                            next_agent=["SoilHealthInfoAgent"],
                        ).json()
                    )
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="SoilHealthInfoAgent", status="Started: soil analysis"
                        ).json()
                    )
                    logger.info(
                        f"soil_prompt_missing: {soil_prompt_missing} {datetime.utcnow().isoformat()}"
                    )

                # Update maps with the results from parallel calls
                if future_env:
                    env_refill: EnvironmentalResponse = future_env.result()
                    for env_data in env_refill.environmental_factors:
                        env_map[env_data.crop] = env_data
                    self.status_callback(
                        AgentStatusUpdate(
                            agent="EnvironmentalInfoAgent",
                            status="Completed task",
                            log=self.log_steps[-1]["response"],
                            next_agent=["VerificationAgent"],
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
                            log=self.log_steps[-1]["response"],
                            next_agent=["VerificationAgent"],
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
                            log=self.log_steps[-1]["response"],
                            next_agent=["VerificationAgent"],
                        ).json()
                    )

            retry += 1
            time.sleep(1)  # Optional delay between retries

        # After max retries, for any crop still missing data, set fallback values.
        for crop in crop_list:
            # Re-run verification to check for missing domains one last time.
            combined = {
                "crop": crop,
                "environmental": (
                    env_map.get(crop, {}).dict() if crop in env_map else {}
                ),
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
                        additional_info=self.FALLBACK_MESSAGE,
                    )
                    logger.info(f"Fallback for environmental data for {crop}")
                if "marketing" in missing and crop not in mkt_map:
                    mkt_map[crop] = MarketingData(
                        crop=crop,
                        pricing_trends=self.FALLBACK_MESSAGE,
                        demand_forecast=self.FALLBACK_MESSAGE,
                        distribution_channels=self.FALLBACK_MESSAGE,
                        economic_viability=self.FALLBACK_MESSAGE,
                    )
                    logger.info(f"Fallback for marketing data for {crop}")
                if "soil_health" in missing and crop not in sh_map:
                    sh_map[crop] = SoilHealthData(
                        crop=crop,
                        soil_ph=0,
                        nutrient_levels=self.FALLBACK_MESSAGE,
                        water_retention=self.FALLBACK_MESSAGE,
                        organic_matter=self.FALLBACK_MESSAGE,
                        additional_info=self.FALLBACK_MESSAGE,
                    )
                    logger.info(f"Fallback for soil health data for {crop}")
        if stopVerification == False:
            logger.info(
                f"Final verification loop completed with missing data: {missing_env}, {missing_mkt}, {missing_soil}"
            )
            self.status_callback(
                AgentStatusUpdate(
                    agent="VerificationAgent",
                    status="Completed task",
                    log=self.log_steps[-1]["response"],
                    next_agent=["SummaryAgent"],
                ).json()
            )

        final_structured: Dict[str, Dict] = {}
        for crop in crop_list:
            env = env_map.get(crop)
            mkt = mkt_map.get(crop)
            sh = sh_map.get(crop)

            try:
                final_structured[crop] = {
                    "environmental": env.dict() if env else {},
                    "marketing": mkt.dict() if mkt else {},
                    "soil_health": sh.dict() if sh else {},
                }
            except Exception as e:
                logger.error(f"Error building structured data for {crop}: {e}")
                final_structured[crop] = {
                    "environmental": {
                        "crop": crop,
                        "climate_suitability": getattr(
                            env, "climate_suitability", self.FALLBACK_MESSAGE
                        ),
                        "rainfall": getattr(env, "rainfall", self.FALLBACK_MESSAGE),
                        "temperature_min": getattr(env, "temperature_min", 0),
                        "temperature_max": getattr(env, "temperature_max", 0),
                        "temperature_optimal": getattr(env, "temperature_optimal", 0),
                        "additional_info": getattr(
                            env, "additional_info", self.FALLBACK_MESSAGE
                        ),
                    },
                    "marketing": {
                        "crop": crop,
                        "pricing_trends": getattr(
                            mkt, "pricing_trends", self.FALLBACK_MESSAGE
                        ),
                        "demand_forecast": getattr(
                            mkt, "demand_forecast", self.FALLBACK_MESSAGE
                        ),
                        "distribution_channels": getattr(
                            mkt, "distribution_channels", self.FALLBACK_MESSAGE
                        ),
                        "economic_viability": getattr(
                            mkt, "economic_viability", self.FALLBACK_MESSAGE
                        ),
                    },
                    "soil_health": {
                        "crop": crop,
                        "soil_ph": getattr(sh, "soil_ph", 0),
                        "nutrient_levels": getattr(
                            sh, "nutrient_levels", self.FALLBACK_MESSAGE
                        ),
                        "water_retention": getattr(
                            sh, "water_retention", self.FALLBACK_MESSAGE
                        ),
                        "organic_matter": getattr(
                            sh, "organic_matter", self.FALLBACK_MESSAGE
                        ),
                        "additional_info": getattr(
                            sh, "additional_info", self.FALLBACK_MESSAGE
                        ),
                    },
                }

        self.status_callback(
            AgentStatusUpdate(
                agent="SummaryAgent", status="Started summarization"
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
 

        self.status_callback(
            AgentStatusUpdate(
                agent="SummaryAgent",
                status="Completed task",
                log=self.log_steps[-1]["response"],
            ).json()
        )

        self.status_callback(WorkflowCompleted(result=summary_report.report).json())

        return RunResponse(summary_report.report, RunEvent.workflow_completed)


app = FastAPI()


@app.websocket("/ws")
async def websocket_crop_planner(websocket: WebSocket):
    await websocket.accept()

    agent_statuses: Dict[str, dict] = {
        "CropRecommenderAgent": {
            "id": str(uuid.uuid4()),
            "name": "Crop Recommender Agent",
            "status": "idle",
        },
        "EnvironmentalInfoAgent": {
            "id": str(uuid.uuid4()),
            "name": "Environmental Information Agent",
            "status": "idle",
        },
        "MarketingInfoAgent": {
            "id": str(uuid.uuid4()),
            "name": "Marketing Information Agent",
            "status": "idle",
        },
        "SoilHealthInfoAgent": {
            "id": str(uuid.uuid4()),
            "name": "Soil Information Agent",
            "status": "idle",
        },
        "VerificationAgent": {
            "id": str(uuid.uuid4()),
            "name": "Verification Agent",
            "status": "idle",
        },
        "SummaryAgent": {
            "id": str(uuid.uuid4()),
            "name": "Summary Agent",
            "status": "idle",
        },
    }

    loop = asyncio.get_running_loop()

    try:
        initial_status = {
            "event": "initial_status",
            "agents": list(agent_statuses.values()),
        }
        await websocket.send_text(json.dumps(initial_status))

        global_workflow = CropPlannerWorkflow(
            session_id=f"crop-planner-{uuid.uuid4()}", storage=None
        )

        while True:
            for key in agent_statuses:
                agent_statuses[key]["status"] = "idle"

            await websocket.send_text(
                json.dumps(
                    {"event": "agents_reset", "agents": list(agent_statuses.values())}
                )
            )

            query = await websocket.receive_text()
            print(f"Received query: {query}")

            if not query or query.strip() == "":
                await websocket.send_text(
                    json.dumps(
                        {
                            "event": "error",
                            "message": "Empty query received. Please provide a location.",
                        }
                    )
                )
                continue

            def send_update(message: str):
                try:
                    update_data = json.loads(message)
                    if update_data.get("event") == "agent_status_update":
                        agent_key = update_data.get("agent")
                        if agent_key in agent_statuses:
                            agent_statuses[agent_key]["status"] = update_data.get(
                                "status"
                            )
                            agent_statuses[agent_key]["log"] = update_data.get("log")
                            agent_statuses[agent_key]["next_agent"] = update_data.get(
                                "next_agent", []
                            )
                            agent_statuses[agent_key]["timestamp"] = update_data.get(
                                "timestamp"
                            )
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_text(
                                    json.dumps(
                                        {
                                            "event": "agent_status_update",
                                            "agent": agent_statuses[agent_key],
                                        }
                                    )
                                ),
                                loop,
                            )
                    elif update_data.get("event") == "workflow_completed":
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text(
                                json.dumps(
                                    {
                                        "event": "workflow_completed",
                                        "result": update_data.get("result"),
                                    }
                                )
                            ),
                            loop,
                        )
                    else:
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text(message), loop
                        )
                except Exception as e:
                    logger.error(f"Error processing status update: {e}")

            wf_msg = WorkflowMessage(
                message=f"Running Crop Planner workflow for: {query}"
            ).json()
            await websocket.send_text(wf_msg)

            global_workflow.question = query
            global_workflow.status_callback = send_update

            try:
                result_future = run_in_threadpool(global_workflow.run)
                result = await asyncio.wait_for(result_future, timeout=600)
            except asyncio.TimeoutError:
                logger.error("Workflow execution timed out after 10 minutes")
                await websocket.send_text(
                    json.dumps(
                        {
                            "event": "error",
                            "message": "Workflow timed out after 10 minutes. Please try a more specific location or fewer crops.",
                        }
                    )
                )
            except Exception as e:
                logger.error(f"Error running workflow: {str(e)}")
                await websocket.send_text(
                    json.dumps(
                        {
                            "event": "error",
                            "message": f"An error occurred while processing your request: {str(e)[:200]}",
                        }
                    )
                )

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        try:
            await websocket.send_text(
                json.dumps(
                    {"event": "error", "message": f"An error occurred: {str(e)[:200]}"}
                )
            )
        except:
            pass
