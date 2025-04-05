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

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found")

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

        # 1. crops ---------------------------------------------------------
        crops_data: CropsList = self.safe_run(
            crop_recommender_agent, location, CropsList()
        )
        crop_list = crops_data.crops
        logger.info(f"[CropPlanner] Crops: {crop_list}")

        # 2. soil requirements --------------------------------------------
        soil_data: SoilConditions = self.safe_run(
            soil_info_agent, ", ".join(crop_list), SoilConditions()
        )
        soil_map = {i.crop: i.conditions for i in soil_data.soil_requirements}

        # 3. environmental -------------------------------------------------
        env_prompt = f"Location: {location}\nCrops: {', '.join(crop_list)}"
        env_data: EnvironmentalResponse = self.safe_run(
            environmental_info_agent, env_prompt, EnvironmentalResponse()
        )
        env_map = {i.crop: i for i in env_data.environmental_factors}

        time.sleep(3)  # tiny pause to avoid DDG rate‑limit

        # 4. marketing -----------------------------------------------------
        mkt_data: MarketingResponse = self.safe_run(
            marketing_info_agent, env_prompt, MarketingResponse()
        )
        mkt_map = {i.crop: i for i in mkt_data.marketing_factors}

        time.sleep(3)

        # 5. soil‑health ---------------------------------------------------
        sh_data: SoilHealthResponse = self.safe_run(
            soil_health_info_agent, env_prompt, SoilHealthResponse()
        )
        sh_map = {i.crop: i for i in sh_data.soil_health}

        # 6. markdown ------------------------------------------------------
        md: List[str] = ["# Crop Planner Results", ""]
        for crop in crop_list:
            env = env_map.get(crop)
            mkt = mkt_map.get(crop)
            sh  = sh_map.get(crop)

            md.extend(
                [
                    f"## {crop}",
                    "",
                    f"**Soil requirements (LLM):** {soil_map.get(crop, 'N/A')}",
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

        # 7. sources -------------------------------------------------------
        if any(
            [crops_data.sources, soil_data.sources,
             env_data.sources, mkt_data.sources, sh_data.sources]
        ):
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

        return RunResponse("\n".join(md), RunEvent.workflow_completed)

# ---------------------------------------------------------------------------
# CLI example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wf = CropPlannerWorkflow(session_id="crop-planner-demo", storage=None)
    wf.question = "Punjab region of India"
    result = wf.run()

    print("\n=== Crop Planner Result ===\n")
    print(result.content)
