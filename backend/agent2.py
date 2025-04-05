# ---------------------------------------------------------------------------
# Crop Planner Agents & Workflow (DuckDuckGo‑only) – *fixed version*
# ---------------------------------------------------------------------------
# Dependencies: agno, python‑dotenv, pydantic, openai, duckduckgo‑search
# ---------------------------------------------------------------------------

import os
from textwrap import dedent
from typing import List

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
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class CropsList(BaseModel):
    """Structured result for Agent‑1 (crop recommendations)."""
    crops: List[str] = Field(default_factory=list, description="List of crops suitable for the region")
    sources: List[str] = Field(default_factory=list, description="Reference URLs for the crop list")


class SoilRequirement(BaseModel):
    """One entry per crop with its ideal soil conditions."""
    crop: str = Field(..., description="Crop name")
    conditions: str = Field(..., description="Ideal soil conditions in 1‑3 sentences")


class SoilConditions(BaseModel):
    """Structured result for Agent‑2 (soil requirements)."""
    soil_requirements: List[SoilRequirement] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list, description="Reference URLs for the soil information")

# ---------------------------------------------------------------------------
# Agent‑1 : Crop Recommender
# ---------------------------------------------------------------------------

crop_recommender_agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key, max_tokens=1500),
    tools=[DuckDuckGoTools()],
    description=dedent(
        """\
        You are **CropRecommenderBot**.
        The user asks which crops can be grown in a particular region.
        Return **only** the list of suitable crops, backed by sources.
        """
    ),
    instructions=dedent(
        """\
        1. Perform DuckDuckGo searches focused on agronomic guidance for the region.
        2. Extract a concise list (5‑10) of commonly recommended crops.
        3. Respond with JSON that matches the `CropsList` schema, e.g.:
           {
             "crops": ["Wheat", "Barley", "Canola"],
             "sources": ["https://…", "https://…"]
           }
        4. The `crops` field must contain only crop names.
        5. Provide up to 5 high‑quality source URLs.
        """
    ),
    response_model=CropsList,
    structured_outputs=True,          # or response_format="json_schema"
    show_tool_calls=True,             # 👈 shows each DuckDuckGo search
    debug_mode=True  # ← new
)

# ---------------------------------------------------------------------------
# Agent‑2 : Soil Information Fetcher
# ---------------------------------------------------------------------------

soil_info_agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key, max_tokens=2500),
    tools=[DuckDuckGoTools()],
    description="You are **SoilInfoBot**. Provide ideal soil requirements for each crop.",
    instructions=dedent(
        """\
        1. For each crop in the supplied comma‑separated list, search DuckDuckGo
           for ideal soil conditions (pH, texture, drainage, fertility, organic matter).
        2. Summarize the conditions in 1‑3 sentences.
        3. Respond with JSON that matches the `SoilConditions` schema, e.g.:
           {
             "soil_requirements": [
               { "crop": "Wheat",
                 "conditions": "Well‑drained loam with pH 6.0‑7.0…" },
               { "crop": "Rice",
                 "conditions": "Clayey, water‑retentive soil, pH 5.5‑6.5…" }
             ],
             "sources": ["https://…", "https://…"]
           }
        4. Include up to 10 reference URLs.
        """
    ),
    response_model=SoilConditions,
    structured_outputs=True,
    show_tool_calls=True,
    debug_mode=True   # ← new
)

# ---------------------------------------------------------------------------
# Workflow : Crop Planner
# ---------------------------------------------------------------------------

class CropPlannerWorkflow(Workflow):
    """Ask which crops grow in a region, then fetch soil requirements for each crop."""

    description: str = (
        "Ask which crops grow in a region, then fetch soil requirements for each crop."
    )

    def run(self) -> RunResponse:
        # Validate user input
        user_question: str | None = getattr(self, "question", None)
        if not user_question:
            raise ValueError("Please set `workflow.question` before calling run().")

        logger.info(f"[CropPlannerWorkflow] Starting for query: {user_question}")

        # Step‑1 : Crop recommendation
        crop_response = crop_recommender_agent.run(user_question)
        crops_data: CropsList = crop_response.content
        crop_list = crops_data.crops
        logger.info(f"[CropPlannerWorkflow] Recommended crops: {crop_list}")

        # Step‑2 : Soil requirements
        soil_prompt = ", ".join(crop_list)            # SoilInfoBot expects a comma‑separated list
        soil_response = soil_info_agent.run(soil_prompt,stream=True,show_full_reasoning=True)
        soil_data: SoilConditions = soil_response.content

        # Assemble final Markdown
        md_lines: List[str] = ["## Recommended Crops", ""]
        for crop in crop_list:
            md_lines.append(f"- {crop}")

        md_lines.extend(["", "## Soil Requirements", ""])
        for item in soil_data.soil_requirements:
            md_lines.append(f"**{item.crop}**: {item.conditions}")

        # Sources
        if crops_data.sources or soil_data.sources:
            md_lines.extend(["", "### Sources"])
            if crops_data.sources:
                md_lines.append("**Crop recommendation sources:**")
                md_lines.extend(f"- {s}" for s in crops_data.sources)
            if soil_data.sources:
                md_lines.append("**Soil requirement sources:**")
                md_lines.extend(f"- {s}" for s in soil_data.sources)

        return RunResponse(
            content="\n".join(md_lines),
            event=RunEvent.workflow_completed,
        )

# ---------------------------------------------------------------------------
# Example CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    question = "What crops can be grown in the Punjab region of India?"

    workflow = CropPlannerWorkflow(session_id="crop-planner-demo", storage=None)
    workflow.question = question
    result = workflow.run()

    print("\n=== Crop Planner Result ===\n")
    print(result.content)
