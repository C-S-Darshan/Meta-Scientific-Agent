import { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import './index.css';

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [history, setHistory] = useState([]);
  const [currentChat, setCurrentChat] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark');

  // Apply theme to root element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => { fetchHistory(); }, []);

  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/history`);
      setHistory(response.data.history);
    } catch (error) {
      console.error("Failed to fetch history", error);
    }
  };

  const toggleSidebar = () => setIsSidebarCollapsed(!isSidebarCollapsed);
  const toggleTheme  = () => setTheme(t => t === 'dark' ? 'light' : 'dark');
  const handleNewChat = () => setCurrentChat(null);

  const handleDeleteHistory = async (reportId) => {
    try {
      await axios.delete(`${API_BASE_URL}/history/${reportId}`);
      setHistory(prev => prev.filter(h => h.report_id !== reportId));
      // If the deleted report is currently being viewed, clear it
      if (currentChat?.report_id === reportId) {
        setCurrentChat(null);
      }
    } catch (error) {
      console.error("Failed to delete report", error);
    }
  };

  const handleSelectHistory = (item) => {
    // Reconstruct pipeline steps from persisted RAG context
    const steps = [];
    const ctx = item.rag_context;
    if (ctx) {
      // Baseline alloy step
      if (ctx.baseline) {
        const alloyId = ctx.baseline.includes('[') 
          ? ctx.baseline.split(' | ')[0].replace('[', '') 
          : 'Alloy';
        steps.push({
          id: 'rag_alloy', status: 'done',
          label: `Baseline retrieved: ${alloyId}`,
          detail: ctx.baseline,
        });
      }
      // Metallurgical rule step
      if (ctx.rule) {
        const ruleParts = ctx.rule.replace(/[\[\]]/g, '').split(' | ');
        const element = ruleParts[0] || 'element';
        const goalStr = ruleParts.length > 1 ? ruleParts[1].replace('goal: ', '') : '';
        steps.push({
          id: 'rag_rule', status: 'done',
          label: `Rule matched: ${element} → ${goalStr}`,
          detail: ctx.rule,
        });
      }
      // Application context step
      if (ctx.app_context && ctx.app_context !== 'None') {
        let appLabel = 'General';
        for (const part of ctx.app_context.split('], [')) {
          if (part.toLowerCase().includes('application |')) {
            appLabel = part.split('| application |').pop().trim().replace(/[\[\]]/g, '');
            break;
          }
        }
        steps.push({
          id: 'rag_app', status: 'done',
          label: `Application context: ${appLabel}`,
          detail: ctx.app_context,
        });
      }
    }

    setCurrentChat({
      report_id: item.report_id,
      prompt: item.goal,
      markdown: item.markdown,
      chart_urls: item.chart_urls || [],
      thermo_urls: item.thermo_urls || [],
      steps,
    });
  };

  const handleSendMessage = async (prompt, maxIterations = 1) => {
    setIsLoading(true);
    setCurrentChat({ prompt, isPending: true, steps: [] });

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_iterations: maxIterations }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));

            if (event.type === 'step') {
              setCurrentChat(prev => {
                const steps = [...(prev?.steps ?? [])];
                const idx = steps.findIndex(s => s.id === event.id);
                const stepObj = { id: event.id, status: event.status, label: event.label, detail: event.detail, duration: event.duration };
                if (idx >= 0) steps[idx] = stepObj;
                else steps.push(stepObj);
                return { ...prev, steps };
              });
            } else if (event.type === 'result') {
              setCurrentChat(prev => ({
                report_id: event.report_id,
                prompt,
                markdown: event.markdown,
                chart_urls:  event.chart_urls  || [],
                thermo_urls: event.thermo_urls || [],
                steps: prev?.steps || [],
              }));
              fetchHistory();
            } else if (event.type === 'error') {
              setCurrentChat(prev => ({
                ...prev,
                isPending: false,
                markdown: `**Error:** ${event.message}`,
                steps: prev?.steps ?? [],
              }));
            }
          } catch (_) { /* malformed JSON — skip */ }
        }
      }
    } catch (error) {
      console.error("Stream error:", error);
      setCurrentChat({ prompt, markdown: `**Error:** ${error.message}`, steps: [] });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Sidebar
        isCollapsed={isSidebarCollapsed}
        history={history}
        onNewChat={handleNewChat}
        onSelectHistory={handleSelectHistory}
        onDeleteHistory={handleDeleteHistory}
      />
      <ChatArea
        currentChat={currentChat}
        isLoading={isLoading}
        theme={theme}
        onToggleTheme={toggleTheme}
        onSendMessage={handleSendMessage}
        onToggleSidebar={toggleSidebar}
        isSidebarCollapsed={isSidebarCollapsed}
      />
    </div>
  );
}

export default App;
