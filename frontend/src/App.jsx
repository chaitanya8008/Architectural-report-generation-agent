import React, { useState, useEffect, useRef } from 'react';
import Markdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Cpu, Search, Activity, Database, Clock, ChevronRight, X, Info, AlertTriangle, FolderOpen } from 'lucide-react';
import './App.css';

const API_BASE = 'https://potbelly-cuplike-mariam.ngrok-free.dev';

class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false }; }
  static getDerivedStateFromError(error) { return { hasError: true }; }
  render() {
    if (this.state.hasError) return <div className="error-fallback"><AlertTriangle size={14} /> Rendering Error</div>;
    return this.props.children;
  }
}

const QUICK_ACTIONS = [
  { title: "Project Overview", desc: "Summarize the Dayton Hotel project status.", prompt: "Can you give me a general overview of the project?" },
  { title: "Wall Ratings", desc: "Find STC requirements for guest rooms.", prompt: "What are the STC requirements for guest room walls?" },
  { title: "Drawing Search", desc: "Summarize contents of sheet A0.02.", prompt: "What information is on sheet A0.02?" },
  { title: "Acoustic Audit", desc: "Check for discrepancies in the report.", prompt: "Are there any discrepancies in the acoustic report?" }
];

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState('fast'); // 'fast' or 'pro'
  const [stats, setStats] = useState({ chunks: 113798, latency: "0.43s", status: "Ready", index_date: "2026-05-09" });
  const [threadId] = useState(() => Math.random().toString(36).substring(7));
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState('');
  
  const [currentAssistantMessage, setCurrentAssistantMessage] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [lastThought, setLastThought] = useState('');
  const [toolSteps, setToolSteps] = useState([]);
  const [sources, setSources] = useState([]);
  const [activeSource, setActiveSource] = useState(null);
  
  const scrollRef = useRef(null);

  useEffect(() => {
    const headers = { "ngrok-skip-browser-warning": "true" };
    fetch(`${API_BASE}/stats`, { headers }).then(r => r.json()).then(d => setStats(s => ({...s, ...d}))).catch(() => {});
    fetch(`${API_BASE}/projects`, { headers }).then(r => r.json()).then(d => {
      if (d.projects && d.projects.length > 0) {
        setProjects(d.projects);
        setSelectedProject(d.projects[0].project_id);
      }
    }).catch(() => {});
  }, []);

  const forceScrollToBottom = () => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  };

  // Smart auto-scroll: only scroll down if the user is already near the bottom
  useEffect(() => {
    if (scrollRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 200;
      if (isNearBottom) {
        scrollRef.current.scrollTop = scrollHeight;
      }
    }
  }, [messages, currentAssistantMessage, lastThought, toolSteps]);

  const handleSubmit = async (e, customPrompt) => {
    if (e && e.preventDefault) e.preventDefault();
    const userMessage = typeof e === 'string' ? e : (customPrompt || input.trim());
    if (!userMessage || isLoading) return;

    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setTimeout(forceScrollToBottom, 50);
    setIsLoading(true);
    setIsThinking(true);
    setToolSteps([]);
    setSources([]);
    setLastThought('');
    setCurrentAssistantMessage('');

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({ 
          message: userMessage, 
          thread_id: threadId,
          mode: mode,
          project_id: selectedProject 
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = '';
      let tempSources = [];

      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        // Decode with stream:true to handle split characters
        buffer += decoder.decode(value, { stream: true });
        
        // SSE events are separated by double newlines
        const parts = buffer.split(/\r?\n\r?\n/);
        buffer = parts.pop(); // Keep incomplete event in buffer

        for (const part of parts) {
          const lines = part.split(/\r?\n/);
          let eventType = '';
          let eventData = [];

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              // Extract data, removing the single optional leading space
              const dataStr = line.startsWith('data: ') ? line.slice(6) : line.slice(5);
              eventData.push(dataStr);
            }
          }

          if (!eventType) continue;
          const data = eventData.join('\n');

          if (eventType === 'text') {
            setIsThinking(false);
            assistantText += data;
            setCurrentAssistantMessage(assistantText);
          } else if (eventType === 'status') {
            setLastThought(data); // Reuse thought field for status updates
            setIsThinking(true);
          } else if (eventType === 'thought') {
            setLastThought(data);
          } else if (eventType === 'tool') {
            try {
              const tool = JSON.parse(data);
              setToolSteps(prev => prev.some(s => s.id === tool.id) ? prev : [...prev, tool]);
            } catch (err) {}
          } else if (eventType === 'source') {
            try {
              const src = JSON.parse(data);
              tempSources.push(src);
              setSources([...tempSources]);
            } catch (err) {}
          } else if (eventType === 'done') {
            setMessages(prev => [...prev, { role: 'assistant', content: assistantText, sources: [...tempSources] }]);
            setCurrentAssistantMessage('');
            setIsThinking(false);
            setLastThought('');
            setToolSteps([]);
          }
        }
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: `⚠️ Error: ${error.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderContent = (content, msgSources = []) => {
    if (!content || typeof content !== 'string') return null;

    try {
      // Split by Bold (**text**), Citations ([n]), or Newlines (\n)
      const parts = content.split(/(\*\*.*?\*\*|\[\d+(?:,\s*\d+)*\]|\n)/g);
      const elements = [];

      parts.forEach((part, i) => {
        if (!part) return;

        // Handle Newlines
        if (part === '\n') {
          elements.push(<br key={`br-${i}`} />);
          return;
        }

        // Handle Bold Text
        if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
          elements.push(<strong key={`bold-${i}`}>{part.slice(2, -2)}</strong>);
          return;
        }

        // Handle Citations
        const citeMatch = part.match(/^\[(\d+(?:,\s*\d+)*)\]$/);
        if (citeMatch) {
          const nums = citeMatch[1].split(',').map(n => n.trim());
          nums.forEach((n, idx) => {
            const num = parseInt(n);
            const sourceList = Array.isArray(msgSources) && msgSources.length > 0 ? msgSources : (Array.isArray(sources) ? sources : []);
            const source = sourceList[num - 1];
            
            elements.push(
              <span 
                key={`cite-${i}-${idx}`} 
                className="citation-badge" 
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setActiveSource(source || { content: `Source snippet #${num} is being synchronized...` });
                }}
              >
                {num}
              </span>
            );
          });
          return;
        }

        // Handle Regular Text
        elements.push(<span key={`text-${i}`}>{part}</span>);
      });

      return (
        <ErrorBoundary>
          <div className="prose" style={{ color: '#ffffff', width: '100%' }}>
            {elements}
          </div>
        </ErrorBoundary>
      );
    } catch (err) {
      console.error("Custom Parser Error:", err);
      return <div className="prose">{content}</div>;
    }
  };

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-group">
          <div className="sidebar-title">Intelligence Panel</div>
          <div className="stats-list" style={{marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem'}}>
            <div className="stat-card">
              <span className="stat-label"><Database size={14} /> Chunks</span>
              <span className="stat-val">{stats.chunks.toLocaleString()}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label"><Clock size={14} /> Latency</span>
              <span className="stat-val">{stats.latency}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label"><Activity size={14} /> Status</span>
              <span className="stat-val" style={{color: '#10b981'}}>{stats.status}</span>
            </div>
          </div>
        </div>
        
        <AnimatePresence>
          {activeSource && (
            <motion.div className="source-inspector" initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: -20, opacity: 0 }}>
              <div className="inspector-header">
                <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.7rem', color: 'var(--accent)', fontWeight: 700}}>
                  <Info size={12} /> VERIFIED SOURCE
                </div>
                <X size={14} style={{cursor: 'pointer'}} onClick={() => setActiveSource(null)} />
              </div>
              <div className="inspector-content">{activeSource.content}</div>
            </motion.div>
          )}
        </AnimatePresence>
      </aside>

      <main className="main-content">
        <div className="chat-container">
          <header className="header">
            <div className="logo-group">
              <div className={`logo-icon ${isThinking ? 'pulsing' : ''}`}>
                <Cpu size={20} color="#fff" />
              </div>
              <div className="logo-text">AcoustiQ <span style={{opacity: 0.4, fontWeight: 400}}>PRO</span></div>
            </div>

            <div className="header-controls">
              {projects.length > 0 && (
                <div className="project-selector">
                  <FolderOpen size={14} className="project-icon" />
                  <select
                    value={selectedProject}
                    onChange={(e) => setSelectedProject(e.target.value)}
                    className="project-dropdown"
                  >
                    {projects.map(p => (
                      <option key={p.project_id} value={p.project_id}>{p.display_name}</option>
                    ))}
                  </select>
                </div>
              )}

              <div className="mode-toggle">
                <button 
                  className={`mode-btn ${mode === 'fast' ? 'active' : ''}`}
                  onClick={() => setMode('fast')}
                >
                  <Activity size={14} /> Fast
                </button>
                <button 
                  className={`mode-btn ${mode === 'pro' ? 'active' : ''}`}
                  onClick={() => setMode('pro')}
                >
                  <Bot size={14} /> Pro
                </button>
              </div>
            </div>
          </header>

          <div className="chat-window" ref={scrollRef}>
            <ErrorBoundary>
              <AnimatePresence mode="wait">
                {messages.length === 0 ? (
                  <motion.div key="welcome" className="welcome-container" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <div className="welcome-text"><h1 className="welcome-title">Mission Control</h1><p className="welcome-subtitle">Acoustic Consultant AI. Project Intelligence Engine.</p></div>
                    <div className="quick-actions">
                      {QUICK_ACTIONS.map((action, i) => (
                        <div key={i} className="action-card" onClick={() => handleSubmit(action.prompt)}>
                          <div style={{display: 'flex', justifyContent: 'space-between'}}><h3>{action.title}</h3><ChevronRight size={16} opacity={0.3} /></div>
                          <p>{action.desc}</p>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="chat-history" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    {messages.map((msg, idx) => (
                      <div key={idx} className={`message-row ${msg.role}`}>
                        <div className={`avatar ${msg.role}`}>{msg.role === 'assistant' ? <Bot size={18} /> : <User size={18} />}</div>
                        <div className="message-content">{renderContent(msg.content, msg.sources)}</div>
                      </div>
                    ))}
                    {(isThinking || currentAssistantMessage) && (
                      <div className="message-row assistant">
                        <div className="avatar assistant"><Bot size={18} /></div>
                        <div className="message-content">
                          {isThinking && (
                            <div className="process-indicator">
                              <div className="process-line"><div className="process-line-active"></div></div>
                              <div className="process-info"><span>Thinking</span>{toolSteps.length > 0 && <span className="tool-badge"><Search size={10} /> {toolSteps[toolSteps.length-1].name}</span>}</div>
                              {lastThought && <div className="thought-snippet">{lastThought}</div>}
                            </div>
                          )}
                          {currentAssistantMessage && renderContent(currentAssistantMessage)}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </ErrorBoundary>
          </div>

          <form className="input-area" onSubmit={handleSubmit}><div className="input-container">
              <textarea placeholder="Ask AcoustiQ Pro..." rows="1" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) handleSubmit(e); }} />
              <button className="send-btn" type="submit" disabled={isLoading || !input.trim()}><Send size={18} /></button>
          </div></form>
        </div>
      </main>
    </div>
  );
}

export default App;
