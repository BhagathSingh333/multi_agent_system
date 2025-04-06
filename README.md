# 🌾 AgroIntelligence — A Multi-Agent System for Smarter Agriculture
 
![AgroIntelligence Banner](https://raw.githubusercontent.com/jgdshkovi/SP-25/265174294186c9b5ec95f7b70a9934d03d6482a3/agent_workflow.jpg)
 
## 🚀 Inspiration
 
Farmers often face difficulty accessing reliable data on weather, soil conditions, and market trends when deciding what crops to grow. Inspired by this, we built **AgroIntelligence**, a multi-agent AI system that simulates the decision-making of agricultural experts — each specializing in a particular domain — to recommend optimal crops in any region based on real-world data.
 
Our goal is to empower data-driven, sustainable agriculture.
 
---
 
## 💡 What it does
 
AgroIntelligence:
- Takes a **location** as input.
- Uses **five specialized agents** to gather and analyze:
  - Crop suitability
  - Environmental data
  - Soil health
  - Market demand/supply
- Synthesizes this information to **recommend ~10 crops**, backed by reasoning and data.
- Displays the real-time thinking process of each agent on a **React dashboard** with **WebSocket updates**.
 
---
 
## 🏗️ How we built it
 
- Built using **Agno**, an open-source multi-agent orchestration framework.
- Agents communicate by **passing structured responses** in JSON format.
- FastAPI serves as the **backend** and agent controller, handling requests and WebSocket streams.
- The **frontend**, built with React, receives agent outputs in real-time and displays them in an interactive UI.
 
---
 
## 🤖 Agent Overview
 
| Agent Name              | Description                                                             |
|------------------------|--------------------------------------------------------------------------|
| `CropRecommenderAgent` | Recommends crops suitable for the region using geolocation and agronomy. |
| `EnvironmentalInfoAgent` | Checks climate requirements for the recommended crops.                |
| `MarketingInfoAgent`   | Assesses current demand and market trends.                               |
| `SoilHealthInfoAgent`  | Analyzes soil quality, nutrients, and pH levels.                         |
| `VerificationAgent`    | Makes sure that data given by all the domain Agents is complete and resolves conflicts.|
| `SummaryAgent`         | Synthesizes all results into final recommendation output.                |
 
---
 
## 🧬 Key Code Snippets
 
### 🛠️ Agent Definition
 
```python
crop_recommender_agent = Agent(
    model=make_chat(1000),
    tools=[DuckDuckGoTools()],
    description="CropRecommenderAgent",
    instructions=thinking_prompt(
        """Search DuckDuckGo for agronomic guidance for the region.
        Return 5‑10 suitable crops with sources in `CropsList` format.
        """
    ),
    response_model=CropsList,
    structured_outputs=True
)
```
 
### 🔄 Workflow Setup
 
```python
def run(self) -> RunResponse:
    self.log_steps: List[Dict[str, Any]] = []
    ...
    self.status_callback(
        AgentStatusUpdate(
            agent="CropRecommenderAgent", status="Started crop recommendation"
        ).json()
    )
 
    crops_data: CropsList = self.safe_run(
        crop_recommender_agent, location, CropsList()
    )
    self.status_callback(
        AgentStatusUpdate(
            agent="CropRecommenderAgent",
            status="...
                ...,
            next_agent=["SoilHealthInfoAgent", ...,],
        ).json()
    )
    ...
```
 
### ⚡ Real-time WebSocket Handling (Frontend)
 
```jsx
useEffect(() => {
  // Connect to FastAPI backend
  socketRef.current = new WebSocket("ws://localhost:8000/ws");
 
  socketRef.current.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleWebSocketMessage(data);
  };
  socketRef.current.onerror = (error) => {
      setOutput(prev => prev + "\nError connecting to server");
      setLoading(false);
    };
  socketRef.current.onclose = () => { ... };
  return () => socketRef.current?.close();
  }, []);
```
 
### 📡 FastAPI WebSocket Route
 
```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    user_input = await websocket.receive_text()
    async for update in run_workflow_stream(user_input):
        await websocket.send_text(update.json())
```
 
### 🧠 Architecture Diagram
 
 
### 🧗 Challenges we ran into
- **Chaining Agent Responses:** Passing outputs as structured JSON between agents while keeping context.
 
- **WebSocket Integration:** Real-time updates in React from Python (FastAPI) without delays.
 
- **Structured Output from LLMs:** LLMs often responded with unstructured text; we enforced schemas to fix this.
 
- **Debugging Multi-agent Flows:** Identifying where breakdowns occurred in an agent chain was tricky.
 
 
## 🏅 Accomplishments that we're proud of
 
- Built an end-to-end agent-based AI system that mirrors human decision-making.
- Achieved real-time data aggregation using WebSockets and LLMs.
- Created a highly visual dashboard showing live agent states, interactions, and recommendations.
- Modular agent system — easily extendable to other domains like disaster planning or smart cities.
 
## 📚 What we learned
 
- Deep understanding of multi-agent system design and orchestration.
- Structuring agent communication via schemas and response models.
- WebSocket integration between Python backend and React frontend.
- Techniques for handling LLM unpredictability and timeouts in real-time applications.
 
## 🌱 What's next for AgroIntelligence
 
- Add support for live weather APIs, soil sensors, and market APIs.
- Expand to support personalized recommendations per farmer.
- Translate to regional languages and add voice interface.
- Integrate with blockchain supply chains for provenance tracking.
- Release as a mobile app for wider reach in rural areas.
 
## 🛠️ Tech Stack
 
- **Backend**: Python, Agno, FastAPI, WebSockets
- **Frontend**: React.js, Tailwind CSS
- **Data**: DuckDuckGoTools (search), Open-source LLMs (via Ollama)
- **Architecture**: Multi-Agent Workflow (Agno), REST + Socket APIs
 
## 📂 Project Structure
```graphql
AgroIntelligence/
├── backend/ # FastAPI backend + WebSocket endpoints
│   ├── agent2_v1.py
│   └── requirements.txt
├── agent-dashboard/
│   ├── App.jsx
│   └── components/
│       ├── AgentVisualization.jsx
│       ├── DebugHelper.jsx
│       └── SequentialTreeVisualization.jsx
├── package.json.py
├── index.html
├── ...
├── requirements.txt
└── README.md
```
 
 
## 🧪 Run Locally
 
### 🐍 Backend
 
1. Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
 
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
 
3. Run the FastAPI backend with Uvicorn:
    ```bash
    uvicorn agent2_v1:app --reload
    ```
 
### ⚛️ Frontend
 
1. Navigate to the `agent-dashboard` directory:
    ```bash
    cd agent-dashboard
    ```
 
2. Install the required frontend dependencies:
    ```bash
    npm install
    ```
 
3. Run the React frontend:
    ```bash
    npm run dev
    ```
 
4. Open your browser and navigate to [http://localhost:3000](http://localhost:3000) to view the dashboard.
 
---
 
Made with 🧠+🌱 by **Team "Barbenheimer"** for *Luddy Hackathon* 2025.
 
 
 
 
 
 
