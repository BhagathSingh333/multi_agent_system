# ---------------------------------------------------------------------------
# Crop Planner – resilient version with parallel agent execution
# ---------------------------------------------------------------------------

import os
import time
import random
from textwrap import dedent
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff  # Make sure to pip install backoff
from requests.exceptions import RequestException, Timeout
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
# Pydantic Schemas (unchanged)
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
# Agents with enhanced error handling
# ---------------------------------------------------------------------------

def make_chat(max_tokens: int = 2500):
    return OpenAIChat(id="gpt-4o", api_key=openai_api_key, max_tokens=max_tokens)

# Custom DuckDuckGo tools with retry backoff
class ResilientDuckDuckGoTools(DuckDuckGoTools):
    @backoff.on_exception(
        backoff.expo,
        (RequestException, Timeout),
        max_tries=5,
        max_time=60,
        jitter=backoff.full_jitter
    )
    def search(self, *args, **kwargs):
        return super().search(*args, **kwargs)

# Create agents with resilient tools
def create_agent(description, instructions, response_model):
    return Agent(
        model=make_chat(),
        tools=[ResilientDuckDuckGoTools()],
        description=description,
        instructions=instructions,
        response_model=response_model,
        structured_outputs=True,
    )

crop_recommender_agent = create_agent(
    description="You are **CropRecommenderBot**.",
    instructions=dedent(
        """\
        1. Search DuckDuckGo for agronomic guidance for the region.
        2. Return 5‑10 suitable crops with sources in `CropsList` format.
        """
    ),
    response_model=CropsList,
)

soil_info_agent = create_agent(
    description="You are **SoilInfoBot**.",
    instructions=dedent(
        """\
        For each crop supplied (comma‑separated) search for ideal soil conditions
        and return JSON that matches the `SoilConditions` schema.
        """
    ),
    response_model=SoilConditions,
)

environmental_info_agent = create_agent(
    description="You are **EnvironmentalInfoBot**.",
    instructions=dedent(
        """\
        Input: location + comma‑separated crop list …
        (full instructions unchanged)
        """
    ),
    response_model=EnvironmentalResponse,
)

marketing_info_agent = create_agent(
    description="You are **MarketingInfoBot**.",
    instructions=dedent(
        """\
        Input: location + comma‑separated crop list …
        (full instructions unchanged)
        """
    ),
    response_model=MarketingResponse,
)

soil_health_info_agent = create_agent(
    description="You are **SoilHealthInfoBot**.",
    instructions=dedent(
        """\
        Input: location + comma‑separated crop list …
        (full instructions unchanged)
        """
    ),
    response_model=SoilHealthResponse,
)

# ---------------------------------------------------------------------------
# Workflow with enhanced resilience and parallel execution
# ---------------------------------------------------------------------------

class ParallelCropPlannerWorkflow(Workflow):
    description = (
        "Recommend crops, then gather soil, environmental, marketing and "
        "soil‑health data for each crop in parallel."
    )

    def safe_run(self, agent: Agent, prompt: str, empty: Any) -> Any:
        """Run an agent with retries; return empty model on failure."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                return agent.run(prompt).content
            except Exception as e:
                logger.warning(
                    f"[CropPlanner] {agent.description.split()[1]} failed (attempt {attempt}/{max_attempts}): {e}"
                )
                if attempt < max_attempts:
                    # Add jitter to avoid DDG rate-limiting
                    sleep_time = 5 + random.uniform(1, 5) * attempt
                    logger.info(f"[CropPlanner] Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"[CropPlanner] All attempts failed for {agent.description}.")
                    return empty
    
    def process_batch_parallel(self, crops: List[str], agent: Agent, prompt_template: str, 
                             response_model, extract_func, max_workers: int = 3) -> Dict[str, Any]:
        """Process crops in parallel with controlled concurrency"""
        results = {}
        
        def process_crop(crop):
            prompt = prompt_template.format(crop=crop)
            data = self.safe_run(agent, prompt, response_model())
            return crop, extract_func(data, crop)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_crop = {executor.submit(process_crop, crop): crop for crop in crops}
            
            # Process results as they complete
            for future in as_completed(future_to_crop):
                crop = future_to_crop[future]
                try:
                    crop, result = future.result()
                    if result:
                        results[crop] = result
                except Exception as e:
                    logger.error(f"[CropPlanner] Error processing {crop}: {e}")
        
        return results

    def run(self) -> RunResponse:
        location: str | None = getattr(self, "question", None)
        if not location:
            raise ValueError("Set `workflow.question` first")

        # 1. crops ---------------------------------------------------------
        logger.info(f"[CropPlanner] Getting crop recommendations for {location}...")
        crops_data: CropsList = self.safe_run(
            crop_recommender_agent, location, CropsList()
        )
        crop_list = crops_data.crops
        logger.info(f"[CropPlanner] Found {len(crop_list)} crops: {crop_list}")
        
        if not crop_list:
            # Fallback crops if the search fails completely
            logger.warning("[CropPlanner] No crops found, using fallback list")
            crop_list = ["wheat", "rice", "corn", "potatoes", "soybeans"]

        # 2. soil requirements - with batch processing and controlled concurrency
        logger.info("[CropPlanner] Getting soil requirements...")
        
        # Split the crops into reasonable-sized batches to avoid overwhelming the search API
        batch_size = 3
        soil_map = {}
        
        for i in range(0, len(crop_list), batch_size):
            batch = crop_list[i:i+batch_size]
            batch_prompt = ", ".join(batch)
            
            soil_data: SoilConditions = self.safe_run(
                soil_info_agent, batch_prompt, SoilConditions()
            )
            
            for req in soil_data.soil_requirements:
                soil_map[req.crop] = req.conditions
                
            # Add a pause between batches
            if i + batch_size < len(crop_list):
                sleep_time = 3 + random.uniform(1, 2)
                logger.info(f"[CropPlanner] Pausing for {sleep_time:.1f} seconds between batches...")
                time.sleep(sleep_time)

        # 3-5. Process the remaining data types in parallel
        # Define extraction functions for each data type
        def extract_env(data, crop):
            for factor in data.environmental_factors:
                if factor.crop.lower() == crop.lower():
                    return factor
            return None
            
        def extract_marketing(data, crop):
            for factor in data.marketing_factors:
                if factor.crop.lower() == crop.lower():
                    return factor
            return None
            
        def extract_soil_health(data, crop):
            for factor in data.soil_health:
                if factor.crop.lower() == crop.lower():
                    return factor
            return None
        
        logger.info("[CropPlanner] Processing environmental, marketing, and soil health data in parallel...")
        
        # Set up concurrent execution for all three data types
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks for each data type (these run in parallel)
            env_future = executor.submit(
                self.process_batch_parallel,
                crop_list,
                environmental_info_agent,
                "Location: {location}\nCrops: {{crop}}".format(location=location),
                EnvironmentalResponse,
                extract_env,
                max_workers=2  # Control concurrency within each data type
            )
            
            mkt_future = executor.submit(
                self.process_batch_parallel,
                crop_list,
                marketing_info_agent,
                "Location: {location}\nCrops: {{crop}}".format(location=location),
                MarketingResponse,
                extract_marketing,
                max_workers=2
            )
            
            sh_future = executor.submit(
                self.process_batch_parallel,
                crop_list,
                soil_health_info_agent,
                "Location: {location}\nCrops: {{crop}}".format(location=location),
                SoilHealthResponse,
                extract_soil_health,
                max_workers=2
            )
            
            # Get results as they complete
            try:
                env_map = env_future.result()
                logger.info(f"[CropPlanner] Environmental data complete for {len(env_map)} crops")
            except Exception as e:
                logger.error(f"[CropPlanner] Error getting environmental data: {e}")
                env_map = {}
                
            try:
                mkt_map = mkt_future.result()
                logger.info(f"[CropPlanner] Marketing data complete for {len(mkt_map)} crops")
            except Exception as e:
                logger.error(f"[CropPlanner] Error getting marketing data: {e}")
                mkt_map = {}
                
            try:
                sh_map = sh_future.result()
                logger.info(f"[CropPlanner] Soil health data complete for {len(sh_map)} crops")
            except Exception as e:
                logger.error(f"[CropPlanner] Error getting soil health data: {e}")
                sh_map = {}

        # 6. markdown ------------------------------------------------------
        md: List[str] = ["# Crop Planner Results", ""]
        md.append(f"## Location: {location}")
        md.append("")
        
        for crop in crop_list:
            env = env_map.get(crop)
            mkt = mkt_map.get(crop)
            sh  = sh_map.get(crop)

            md.extend(
                [
                    f"## {crop}",
                    "",
                    f"**Soil requirements:** {soil_map.get(crop, 'N/A')}",
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
        sources_list = []
        if hasattr(crops_data, 'sources'):
            sources_list.extend(crops_data.sources)
            
        md.extend(["---", "### Sources"])
        if sources_list:
            for s in set(sources_list):
                if s and len(s.strip()) > 0:
                    md.append(f"- {s}")
        else:
            md.append("No sources available due to search limitations.")

        return RunResponse("\n".join(md), RunEvent.workflow_completed)

# ---------------------------------------------------------------------------
# CLI example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wf = ParallelCropPlannerWorkflow(session_id="parallel-crop-planner-demo", storage=None)
    wf.question = "Punjab region of India"
    result = wf.run()

    print("\n=== Parallel Crop Planner Result ===\n")
    print(result.content)
