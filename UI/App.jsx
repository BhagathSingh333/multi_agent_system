import { useState, useRef, useEffect } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "@/components/ui/accordion"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Loader2, TerminalSquare } from "lucide-react"
import AgentVisualization from "./components/AgentVisualization"
import SequentialTreeVisualization from './components/SequentialTreeVisualization'
// import DebugHelper from './components/DebugHelper' // Import if you want debugging

function App() {
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showAccordion, setShowAccordion] = useState(false);
  const [agentInteractions, setAgentInteractions] = useState([]);
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [agents, setAgents] = useState({});
  const socketRef = useRef(null);
  const conversationStateRef = useRef({
    lastAgent: null,
    activeAgents: {},
    conversationHistory: []
  });

  // Define agent colors and mapping from backend names to frontend names
  const agentColors = {
    // Original simulator colors
    Planner: '#FF6B6B',
    Executor: '#4ECDC4',
    Researcher: '#45B7D1',
    Critic: '#FFA07A',
    Coordinator: '#F7B801',
    
    // Backend agent full names
    "Crop Recommender Agent": '#FF6B6B',     // Using Planner color
    "Environmental Information Agent": '#45B7D1',   // Using Researcher color
    "Marketing Information Agent": '#4ECDC4',       // Using Executor color
    "Soil Information Agent": '#FFA07A',      // Using Critic color
    "Verification Agent": '#F7B801',        // Using Coordinator color
    "Summary Agent": '#8B5CF6',             // New color for Summary Agent
    
    // Frontend friendly names
    "Crop Recommender": '#FF6B6B',
    "Environmental Info": '#45B7D1',
    "Marketing Info": '#4ECDC4',
    "Soil Health": '#FFA07A',
    "Verification": '#F7B801',
    "Summary": '#8B5CF6',
    
    // User for the final interaction
    "User": '#6366F1',
    
    // Backend agent abbreviated names as fallback
    CropRecommenderAgent: '#FF6B6B',
    EnvironmentalInfoAgent: '#45B7D1',
    MarketingInfoAgent: '#4ECDC4',
    SoilHealthInfoAgent: '#FFA07A',
    VerificationAgent: '#F7B801',
    SummaryAgent: '#8B5CF6',
  };

  // Map backend agent names to frontend friendly names
  const agentNameMapping = {
    "Crop Recommender Agent": "Crop Recommender",
    "Environmental Information Agent": "Environmental Info",
    "Marketing Information Agent": "Marketing Info", 
    "Soil Information Agent": "Soil Health",
    "Verification Agent": "Verification",
    "Summary Agent": "Summary",
    
    // Also include the backend names for fallback
    CropRecommenderAgent: "Crop Recommender",
    EnvironmentalInfoAgent: "Environmental Info",
    MarketingInfoAgent: "Marketing Info",
    SoilHealthInfoAgent: "Soil Health",
    VerificationAgent: "Verification",
    SummaryAgent: "Summary",
  };

  // WebSocket setup for backend integration
  useEffect(() => {
    // Connect to FastAPI backend
    socketRef.current = new WebSocket("ws://localhost:8000/ws");

    socketRef.current.onopen = () => {
      console.log("✅ WebSocket connected to backend");
    };

    socketRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("Received websocket data:", data);
        
        handleWebSocketMessage(data);
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
        setOutput(prev => prev + "\nError processing server response");
        setLoading(false);
      }
    };

    socketRef.current.onerror = (error) => {
      console.error("WebSocket error:", error);
      setOutput(prev => prev + "\nError connecting to server");
      setLoading(false);
    };

    socketRef.current.onclose = () => {
      console.log("❌ WebSocket disconnected");
    };

    return () => socketRef.current?.close();
  }, []);

  // Handle WebSocket messages from backend
  const handleWebSocketMessage = (data) => {
    const { event } = data;

    switch (event) {
      case "initial_status":
        // Initialize agent status when connection established
        if (data.agents) {
          const agentsObj = {};
          data.agents.forEach(agent => {
            agentsObj[agent.name] = agent;
          });
          setAgents(agentsObj);
        }
        break;

      case "agents_reset":
        // Handle agents reset (ignore as it's just for state resetting)
        console.log("Agents reset received");
        break;
        
      case "agent_status_update":
        // Process agent status updates
        if (data.agent) {
          const { agent } = data;
          const agentName = agent.name;
          const friendlyName = agentNameMapping[agentName] || agentName;
          
          // Update output with agent status
          if (agent.status && agent.status !== "idle") {
            setOutput(prev => `${prev}\nAgent ${friendlyName}: ${agent.status}\n`);
          }
          
          // Store previous state to detect changes
          const previousAgent = agents[agentName];
          
          // Create interactions when next_agent changes
          if (agent.next_agent && agent.next_agent.length > 0) {
            // Only create interactions if this is a new update with next_agent
            const isNewInteraction = 
              !previousAgent || 
              !previousAgent.next_agent || 
              previousAgent.next_agent.length === 0 ||
              JSON.stringify(previousAgent.next_agent) !== JSON.stringify(agent.next_agent);
              
            if (isNewInteraction) {
              agent.next_agent.forEach(nextAgent => {
                // Format a simplified thought process from the log if available
                let thoughtProcess = "Processing data...";
                if (agent.log) {
                  try {
                    const logObj = JSON.parse(agent.log);
                    thoughtProcess = logObj.thinking || 
                                    "This agent has completed its analysis and is passing data to the next agent.";
                  } catch (e) {
                    thoughtProcess = agent.log.substring(0, 200) + "...";
                  }
                }
                
                addAgentInteraction(
                  agentName,
                  nextAgent,
                  `Sending data to ${agentNameMapping[nextAgent] || nextAgent}`,
                  thoughtProcess
                );
              });
            }
          }
          
          // Update agents state
          setAgents(prev => ({
            ...prev,
            [agentName]: agent
          }));
        }
        break;

      case "workflow_completed":
        // Process completed workflow
        if (data.result) {
          // Create a final interaction from the last active agent to show completion
          const activeAgents = Object.entries(agents)
            .filter(([_, agent]) => agent.status !== 'idle' && agent.status !== 'error')
            .map(([name]) => name);
            
          if (activeAgents.length > 0) {
            const lastAgent = activeAgents[activeAgents.length - 1];
            addAgentInteraction(
              lastAgent, 
              "User", 
              "Final report completed",
              "The workflow has completed successfully and produced a final report."
            );
          }
          
          setOutput(prev => `${prev}\n\nFinal Report:\n${data.result}`);
          setLoading(false);
        }
        break;

      case "error":
        // Handle errors
        if (data.message) {
          setOutput(prev => `${prev}\nError: ${data.message}\n`);
          setLoading(false);
        }
        break;

      case "workflow_message":
        // Handle workflow messages
        if (data.message) {
          setOutput(prev => `${prev}\n${data.message}`);
        }
        break;

      default:
        // Handle any other message types
        console.log("Unhandled event type:", event);
        if (data && typeof data === "string") {
          setOutput(prev => prev + data);
        }
    }
  };

  // Add a new agent interaction to the state
  const addAgentInteraction = (source, target, message, thoughtProcess) => {
    // Map backend agent names to frontend names if needed
    const sourceName = agentNameMapping[source] || source;
    const targetName = agentNameMapping[target] || target;
    
    // Don't add duplicate interactions
    const isDuplicate = agentInteractions.some(
      interaction => 
        interaction.source === sourceName && 
        interaction.target === targetName && 
        interaction.message === message
    );
    
    if (!isDuplicate) {
      setAgentInteractions(prev => [...prev, {
        source: sourceName,
        target: targetName,
        message,
        thoughtProcess,
        timestamp: new Date().getTime()
      }]);
    }
  };

  const handleViewDeepAnalysis = () => {
    setShowAnalysisModal(true);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setShowAccordion(true);
    setOutput("");
    setAgentInteractions([]);
    setLoading(true);

    // Reset conversation state
    conversationStateRef.current = {
      lastAgent: null,
      activeAgents: {},
      conversationHistory: []
    };

    // Send prompt to the backend
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(prompt);
    } else {
      setOutput("Error: WebSocket not connected. Please refresh the page and try again.");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-pink-50 via-sky-100 to-purple-100 p-6 flex flex-col items-center gap-10">
      <Card className="w-full max-w-2xl backdrop-blur-md bg-white/80 border border-white/40 shadow-2xl rounded-2xl hover:shadow-purple-200 transition-shadow">
        <CardHeader>
          <CardTitle className="text-3xl font-bold text-indigo-800 flex items-center gap-2">
            <span className="text-4xl">🧠</span> Multi-Agent Crop Planner
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="flex flex-col md:flex-row gap-4">
            <Input
              placeholder="Enter a location for crop recommendations"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="rounded-xl border-indigo-300 focus:border-indigo-500"
              required
            />
            <Button
              type="submit"
              className="rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-semibold shadow-md hover:brightness-110 transition-all"
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing
                </>
              ) : (
                "Submit"
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {showAccordion && (
        <div className="w-full max-w-3xl">
          <Accordion type="single" collapsible defaultValue="item-1">
            <AccordionItem value="item-1">
              <div className="rounded-2xl shadow-xl border border-slate-200 bg-white">
                <AccordionTrigger
                  className="flex items-center justify-between w-full px-6 py-4 bg-white text-slate-800 text-lg font-semibold rounded-t-2xl cursor-pointer"
                  style={{ textDecoration: "none" }}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">🧠</span>
                    Agent Response
                  </div>
                </AccordionTrigger>

                <AccordionContent className="bg-slate-100 px-6 py-5 text-slate-900 rounded-b-2xl">
                  <div className="mb-4">
                    <span className="font-bold">User Prompt:</span> {prompt}
                  </div>

                  {agentInteractions.length > 0 && (
                    <div className="mb-6">
                      <h3 className="text-lg font-semibold mb-4 text-indigo-800">Agent Interactions</h3>
                      <div className="bg-white rounded-xl border border-slate-200 shadow-sm">
                        <AgentVisualization interactions={agentInteractions} />
                      </div>
                      <div className="mt-3 flex justify-center">
                        <button
                          className="px-4 py-2 bg-indigo-600 text-white rounded-md shadow-sm hover:bg-indigo-700 transition-colors"
                          onClick={() => handleViewDeepAnalysis()}
                        >
                          View In-depth Agent Analysis
                        </button>
                      </div>
                    </div>
                  )}

                  <div className="border-t border-slate-200 my-6"></div>

                  <div>
                    <h3 className="text-lg font-semibold mb-4 text-indigo-800">Output</h3>
                    {loading ? (
                      <div className="flex items-center gap-2 text-sm text-slate-600">
                        <Loader2 className="animate-spin w-4 h-4" />
                        Waiting for agent response...
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap bg-white p-4 rounded-xl border border-slate-200">
                        {output}
                      </div>
                    )}
                  </div>
                  
                  {/* Uncomment to add debug helper if needed */}
                  {/* <DebugHelper agents={agents} showRawData={true} /> */}
                </AccordionContent>
              </div>
            </AccordionItem>
          </Accordion>
        </div>
      )}

      {/* Modal for the in-depth analysis */}
      {showAnalysisModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-4xl w-full h-[90vh] flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold">In-depth Agent Analysis</h2>
              <button
                className="p-2 rounded-full hover:bg-slate-100"
                onClick={() => setShowAnalysisModal(false)}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Sequential Tree Visualization - Scrollable Canvas */}
            <div className="mb-4">
              <h3 className="text-lg font-semibold mb-3">Sequential Agent Flow</h3>
              <div className="border border-slate-200 rounded-xl overflow-auto" style={{ maxHeight: "50vh" }}>
                <SequentialTreeVisualization
                  interactions={agentInteractions}
                  onThoughtProcessSelect={setSelectedNode}
                />
              </div>
            </div>

            {/* Agent Thought Process Section - Always visible */}
            {selectedNode && (
              <div className="mb-4 bg-white rounded-lg p-4 border border-slate-200">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-bold flex items-center">
                    <span className="w-5 h-5 rounded-full mr-2" style={{ backgroundColor: agentColors[selectedNode.name] }}></span>
                    {selectedNode.name}'s Thought Process
                  </h3>
                  <button
                    className="p-1 rounded-full hover:bg-slate-100"
                    onClick={() => setSelectedNode(null)}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <div className="bg-slate-50 p-4 rounded-md border border-slate-200 whitespace-pre-wrap max-h-[15vh] overflow-auto">
                  {selectedNode.thoughtProcess}
                </div>
              </div>
            )}

            {/* Interaction Statistics - Always visible */}
            <div className="border-t border-slate-200 mt-auto pt-4">
              <h3 className="text-lg font-semibold mb-3">Interaction Statistics</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white p-3 rounded-lg border border-slate-200">
                  <div className="text-sm font-medium text-slate-500">Total Interactions</div>
                  <div className="text-2xl font-bold">{agentInteractions.length}</div>
                </div>
                <div className="bg-white p-3 rounded-lg border border-slate-200">
                  <div className="text-sm font-medium text-slate-500">Unique Agents</div>
                  <div className="text-2xl font-bold">
                    {new Set([
                      ...agentInteractions.map(i => i.source),
                      ...agentInteractions.map(i => i.target)
                    ]).size}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;
