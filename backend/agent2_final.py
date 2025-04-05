# ---------------------------------------------------------------------------
# Crop Planner – resilient + verification loop + summarization
# ---------------------------------------------------------------------------
# pip install agno openai python‑dotenv duckduckgo‑search pydantic
# ---------------------------------------------------------------------------

import os, time, json
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

# ---- Verification Schemas ------------------------------------------------

class VerificationItem(BaseModel):
    crop: str
    missing_domains: List[str]  # e.g. ["environmental", "marketing", "soil_health"]

class VerificationResponse(BaseModel):
    missing_info: List[VerificationItem] = Field(default_factory=list)

# ---- Summarization Schema -------------------------------------------------

class SummaryReport(BaseModel):
    report: str = Field(..., description="A detailed summary report including recommendations and rationale.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chat(max_tokens: int = 3000):
    return OpenAIChat(id="gpt-4o", api_key=openai_api_key, max_tokens=max_tokens)

def thinking_prompt(original: str) -> str:
    """Prepends the 'thinking' requirement to an instruction string."""
    return dedent(
        "1. For the following tasks, store your reasoning in a variable named "
        "`thinking`. Return `thinking` at the beginning of your response.\n"
    ) + original

# ---------------------------------------------------------------------------
# Agents  (detailed instructions remain unchanged)
# ---------------------------------------------------------------------------

crop_recommender_agent = Agent(
    model=make_chat(1500),
    tools=[DuckDuckGoTools()],
    description="You are **CropRecommenderBot**.",
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
    description="You are **EnvironmentalInfoBot**.",
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
    description="You are **MarketingInfoBot**.",
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
    description="You are **SoilHealthInfoBot**.",
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
    description="You are **VerificationBot**.",
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
    description="You are **SummaryBot**.",
    instructions=thinking_prompt(
        """Input: You receive the final structured data for a location along with the aggregated information for each crop.
           Your task is to produce a detailed summary report that:
           1. Reviews the environmental, marketing, and soil‑health information for each crop.
           2. Identifies the best crops to grow in the region along with valid reasons for each choice.
           3. Provides actionable suggestions to farmers about crop selection and cultivation practices based on the data.
           Return JSON matching the `SummaryReport` schema."""
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
        "verification pass, refill missing data, and finally summarize the "
        "results with recommendations for farmers."
    )

    def safe_run(self, agent: Agent, prompt: str, empty):
        """Run an agent; return its .content or a default empty model."""
        try:
            res = agent.run(prompt)
            self.log_steps.append(
                f"### {agent.description.split()[1]}\n**Prompt:**\n```\n{prompt}\n```\n"
                f"**Response:**\n```\n{json.dumps(res.content.dict() if hasattr(res.content,'dict') else res.content, indent=2)}\n```\n"
            )
            return res.content
        except Exception as e:
            logger.warning(f"[CropPlanner] {agent.description.split()[1]} failed: {e}")
            self.log_steps.append(f"### {agent.description.split()[1]} ERROR\n{e}\n")
            return empty

    def run(self) -> RunResponse:
        self.log_steps: List[str] = []  # initialize conversation log

        location: str | None = getattr(self, "question", None)
        if not location:
            raise ValueError("Set `workflow.question` first")

        # -- 1. crop recommendation ---------------------------------------
        crops_data: CropsList = self.safe_run(
            crop_recommender_agent, location, CropsList()
        )
        crop_list = crops_data.crops
        logger.info(f"[CropPlanner] Crops: {crop_list}")

        # -- 2. initial domain calls --------------------------------------
        env_prompt = f"{location}\n{', '.join(crop_list)}"
        env_data: EnvironmentalResponse = self.safe_run(
            environmental_info_agent, env_prompt, EnvironmentalResponse()
        )
        env_map = {d.crop: d for d in env_data.environmental_factors}

        time.sleep(2)

        mkt_data: MarketingResponse = self.safe_run(
            marketing_info_agent, env_prompt, MarketingResponse()
        )
        mkt_map = {d.crop: d for d in mkt_data.marketing_factors}

        time.sleep(2)

        sh_data: SoilHealthResponse = self.safe_run(
            soil_health_info_agent, env_prompt, SoilHealthResponse()
        )
        sh_map = {d.crop: d for d in sh_data.soil_health}

        # -- 3. verification & refill loop --------------------------------
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

            if not ver_resp.missing_info:
                continue

            missing = ver_resp.missing_info[0].missing_domains
            if not missing:
                continue

            logger.info(f"[CropPlanner] {crop} missing {missing} – refetching")
            crop_prompt = f"{location}\n{crop}"
            # Log the specific prompt being passed for each missing domain.
            if "environmental" in missing:
                logger.info(f"[CropPlanner] Refetching environmental info for {crop} with prompt:\n{crop_prompt}")
                time.sleep(1)
                env_refill: EnvironmentalResponse = self.safe_run(
                    environmental_info_agent, crop_prompt, EnvironmentalResponse()
                )
                if env_refill.environmental_factors:
                    env_map[crop] = env_refill.environmental_factors[0]

            if "marketing" in missing:
                logger.info(f"[CropPlanner] Refetching marketing info for {crop} with prompt:\n{crop_prompt}")
                time.sleep(1)
                mkt_refill: MarketingResponse = self.safe_run(
                    marketing_info_agent, crop_prompt, MarketingResponse()
                )
                if mkt_refill.marketing_factors:
                    mkt_map[crop] = mkt_refill.marketing_factors[0]

            if "soil_health" in missing:
                logger.info(f"[CropPlanner] Refetching soil-health info for {crop} with prompt:\n{crop_prompt}")
                time.sleep(1)
                sh_refill: SoilHealthResponse = self.safe_run(
                    soil_health_info_agent, crop_prompt, SoilHealthResponse()
                )
                if sh_refill.soil_health:
                    sh_map[crop] = sh_refill.soil_health[0]

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
       

        # -- 7. Append conversation log -------------------------------------
        md.extend(["## Agent Conversation Log", ""])
        md.extend(self.log_steps)
        
         # -- 5. Summarize the complete information using SummaryBot --------
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

        return RunResponse("\n".join(md), RunEvent.workflow_completed)

# ---------------------------------------------------------------------------
# CLI example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wf = CropPlannerWorkflow(session_id="crop-planner-demo", storage=None)
    wf.question = "what crops are better in Chicago USA?"
    result = wf.run()

    print("\n=== Crop Planner Result ===\n")
    print(result.content)
