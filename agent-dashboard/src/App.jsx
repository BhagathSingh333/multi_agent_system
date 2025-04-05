import { useState } from 'react';

function App() {
  const [prompt, setPrompt] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("User Prompt:", prompt);
  };

  return (
    <div style={{ padding: '30px', fontFamily: 'Arial' }}>
      <h2>Multi-Agent Dashboard</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter your prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          style={{ padding: '10px', fontSize: '16px', width: '350px' }}
          required
        />
        <button type="submit" style={{ padding: '10px', marginLeft: '10px', fontSize: '16px' }}>
          Submit
        </button>
      </form>
    </div>
  );
}

export default App;
