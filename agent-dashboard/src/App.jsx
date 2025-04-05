import { useState } from 'react';

function App() {
  const [prompt, setPrompt] = useState('');
  const [showAccordion, setShowAccordion] = useState(false); // controls visibility
  const [isOpen, setIsOpen] = useState(true);               // controls toggle (open/close)
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    setShowAccordion(true); // Show accordion on submit
    setIsOpen(true);        // Reset to open state on submit
    setLoading(true);

    setTimeout(() => {
      setOutput(`Dummy output for: "${prompt}"`);
      setLoading(false);
    }, 2000);
  };

  return (
    <div style={{ padding: '30px', fontFamily: 'Arial', maxWidth: '600px', margin: 'auto' }}>
      <h2>Multi-Agent Dashboard</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter your prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          style={{ padding: '10px', fontSize: '16px', width: '80%' }}
          required
        />
        <button type="submit" style={{ padding: '10px', marginLeft: '10px', fontSize: '16px' }}>
          Submit
        </button>
      </form>

      {showAccordion && (
        <div style={{ marginTop: '20px', border: '1px solid #ccc', borderRadius: '5px' }}>
          <div
            style={{ padding: '10px', cursor: 'pointer', backgroundColor: '#f1f1f1' }}
            onClick={() => setIsOpen(!isOpen)}
          >
            <strong>{isOpen ? '▼' : '▶'} Agent Response</strong>
          </div>
          {isOpen && (
            <div style={{ padding: '15px' }}>
              <div>
                <strong>User Prompt:</strong> {prompt}
              </div>
              {loading ? (
                <div style={{ marginTop: '10px' }}>Loading response...</div>
              ) : (
                <div style={{ marginTop: '10px' }}>
                  <strong>Output:</strong> {output}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
