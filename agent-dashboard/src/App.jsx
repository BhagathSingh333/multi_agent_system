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
import { Loader2 } from "lucide-react"
import AgentVisualization from "./components/AgentVisualization"
import SequentialTreeVisualization from './components/SequentialTreeVisualization'

function App() {
  const [prompt, setPrompt] = useState("")
  const [output, setOutput] = useState("")
  const [loading, setLoading] = useState(false)
  const [showAccordion, setShowAccordion] = useState(false)
  const [agentInteractions, setAgentInteractions] = useState([])
  const [showAnalysisModal, setShowAnalysisModal] = useState(false)
  const socketRef = useRef(null)
  const conversationStateRef = useRef({
    lastAgent: null,
    activeAgents: {},
    conversationHistory: []
  })

  // WebSocket setup - keep this for future backend integration
  useEffect(() => {
    socketRef.current = new WebSocket("ws://localhost:8000/ws")

    socketRef.current.onopen = () => {
      console.log("✅ WebSocket connected")
    }

    socketRef.current.onmessage = (event) => {
      const newText = event.data;
      setOutput(prev => prev + newText)
      setLoading(false)
    }

    socketRef.current.onerror = (error) => {
      console.error("WebSocket error:", error)
      setLoading(false)
    }

    socketRef.current.onclose = () => {
      console.log("❌ WebSocket disconnected")
    }

    return () => socketRef.current?.close()
  }, [])

  // Function to simulate agent interactions for frontend development
  const simulateAgentInteractions = () => {
    const agents = ["Coordinator", "Planner", "Researcher", "Executor", "Critic"];
    let interactionCount = 0;

    const addRandomInteraction = () => {
      if (interactionCount >= 10) {
        setLoading(false);
        return;
      }

      const sourceIndex = Math.floor(Math.random() * agents.length);
      let targetIndex;
      do {
        targetIndex = Math.floor(Math.random() * agents.length);
      } while (targetIndex === sourceIndex);

      const source = agents[sourceIndex];
      const target = agents[targetIndex];

      const messages = [
        "Requesting information",
        "Sharing analysis results",
        "Delegating subtask",
        "Providing context",
        "Asking for verification",
        "Sending recommendations"
      ];
      const message = messages[Math.floor(Math.random() * messages.length)];

      addAgentInteraction(source, target, message);
      interactionCount++;

      // Generate simulated output text
      const outputTexts = [
        `Agent ${source} is processing the request...`,
        `Agent ${target} is analyzing the data from ${source}...`,
        `Collaboration between ${source} and ${target} is in progress...`,
        `${target} received task from ${source} and is working on it...`
      ];

      setOutput(prev => prev + "\n" + outputTexts[Math.floor(Math.random() * outputTexts.length)]);

      // Schedule next interaction
      setTimeout(addRandomInteraction, Math.random() * 800 + 400);
    };

    // Start the simulation
    setTimeout(addRandomInteraction, 500);
  };

  // Add a new agent interaction to the state
  const addAgentInteraction = (source, target, message) => {
    setAgentInteractions(prev => [...prev, {
      source,
      target,
      message,
      timestamp: new Date().getTime()
    }]);
  }

  const handleViewDeepAnalysis = () => {
    // Implement your deep analysis functionality here
    console.log("Opening in-depth agent analysis");
    setShowAnalysisModal(true);
  };

  const handleSubmit = (e) => {
    e.preventDefault()
    setShowAccordion(true)
    setOutput("")
    setAgentInteractions([])
    setLoading(true)

    // Reset conversation state
    conversationStateRef.current = {
      lastAgent: "Coordinator",
      activeAgents: { "Coordinator": true },
      conversationHistory: []
    };

    // Simulate initial agent setup
    setTimeout(() => {
      addAgentInteraction("Coordinator", "Planner", "Initializing task planning");
      setOutput("Agent Coordinator is initializing the task...\n");

      setTimeout(() => {
        addAgentInteraction("Planner", "Coordinator", "Acknowledging task");
        setOutput(prev => prev + "Agent Planner is acknowledging the task from Coordinator...\n");

        setTimeout(() => {
          addAgentInteraction("Planner", "Researcher", "Requesting background information");
          setOutput(prev => prev + "Agent Planner is requesting background information from Researcher...\n");
          
          setTimeout(() => {
            addAgentInteraction("Researcher", "Planner", "Providing relevant data");
            setOutput(prev => prev + "Agent Researcher is providing relevant data to Planner...\n");

            // Start random interactions after initial setup
            simulateAgentInteractions();
          }, 800);
        }, 700);
      }, 600);
    }, 500);

    // Keep the WebSocket send for future backend integration
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(prompt)
    }
  }

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-pink-50 via-sky-100 to-purple-100 p-6 flex flex-col items-center gap-10">
      <Card className="w-full max-w-2xl backdrop-blur-md bg-white/80 border border-white/40 shadow-2xl rounded-2xl hover:shadow-purple-200 transition-shadow">
        <CardHeader>
          <CardTitle className="text-3xl font-bold text-indigo-800 flex items-center gap-2">
            <span className="text-4xl">🧠</span> Multi-Agent Dashboard
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="flex flex-col md:flex-row gap-4">
            <Input
              placeholder="Enter your prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="rounded-xl border-indigo-300 focus:border-indigo-500"
              required
            />
            <Button
              type="submit"
              className="rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-semibold shadow-md hover:brightness-110 transition-all"
            >
              Submit
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
                </AccordionContent>
              </div>
            </AccordionItem>
          </Accordion>
        </div>
      )}

      {/* Modal for the in-depth analysis */}
      {showAnalysisModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-4xl w-full max-h-[90vh] overflow-auto">
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

            {/* Sequential Tree Visualization */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3">Sequential Agent Flow</h3>
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 overflow-auto" style={{ maxHeight: '60vh' }}>
                <SequentialTreeVisualization interactions={agentInteractions} />
              </div>
            </div>

            {/* Additional analysis content */}
            <div className="border-t border-slate-200 my-4 pt-4">
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
  )
}

export default App
