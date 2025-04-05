import { useState } from "react"
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

function App() {
  const [prompt, setPrompt] = useState("")
  const [output, setOutput] = useState("")
  const [loading, setLoading] = useState(false)
  const [showAccordion, setShowAccordion] = useState(false)

  const handleSubmit = (e) => {
    e.preventDefault()
    setShowAccordion(true)
    setLoading(true)

    setTimeout(() => {
      setOutput(`Dummy output for: "${prompt}"`)
      setLoading(false)
    }, 2000)
  }

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-pink-50 via-sky-100 to-purple-100 p-6 flex flex-col items-center gap-10">
      {/* Header / Prompt Card */}
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

      {/* Agent Response */}
      {showAccordion && (
        <div className="w-full max-w-2xl">
          <Accordion type="single" collapsible>
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
                  <div className="mb-2">
                    <span className="font-bold">User Prompt:</span> {prompt}
                  </div>
                  {loading ? (
                    <div className="flex items-center gap-2 text-sm text-slate-600">
                      <Loader2 className="animate-spin w-4 h-4" />
                      Loading response...
                    </div>
                  ) : (
                    <div className="mt-2">
                      <span className="font-bold">Output:</span> {output}
                    </div>
                  )}
                </AccordionContent>
              </div>
            </AccordionItem>
          </Accordion>
        </div>
      )}
    </div>
  )
}

export default App
