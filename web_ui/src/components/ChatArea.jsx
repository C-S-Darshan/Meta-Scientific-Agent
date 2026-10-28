import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_BASE_URL = 'http://localhost:8000';

// Icon map for each pipeline step
const STEP_ICONS = {
  embed:      '⟳',
  rag_alloy:  '🗄',
  rag_rule:   '📐',
  rag_app:    '🔧',
  hypothesis: '🧬',
  physics:    '⚛',
  heat_treat: '🔥',
  analyst:    '🔬',
  calphad:    '🌡',
  thermo_insights: '🔍',
  report:     '📊',
  evolution:  '🧬',
};

function ThinkingSteps({ steps, isPending }) {
  // For historical reports (isPending is undefined), default to expanded.
  // For live pipelines, start expanded and auto-collapse on completion.
  const isHistorical = isPending === undefined || isPending === null;
  const [isExpanded, setIsExpanded] = useState(isHistorical || isPending);
  
  // Auto-collapse only when a LIVE pipeline transitions from pending→done
  useEffect(() => {
    if (isPending === false && !isHistorical) setIsExpanded(false);
  }, [isPending]);

  const activeStep = steps.find(s => s.status === 'active');
  const totalDuration = steps.reduce((acc, s) => {
    if (s.duration) return acc + parseFloat(s.duration);
    return acc;
  }, 0);

  return (
    <div className="thinking-panel">
      <div 
        className="thinking-header" 
        onClick={() => setIsExpanded(!isExpanded)}
        style={{ cursor: 'pointer' }}
      >
        <span className={isPending ? "thinking-pulse" : "step-check"} style={{marginRight: '8px'}}>{!isPending && '✓'}</span>
        <span className="thinking-title">
          {isPending
            ? (activeStep ? activeStep.label : "Running Discovery Pipeline...")
            : isHistorical
              ? "Retrieved Context from Knowledge Base"
              : `Discovery Pipeline Complete (${totalDuration.toFixed(1)}s)`}
        </span>
        <span style={{ marginLeft: 'auto', fontSize: '12px', color: 'var(--text-tertiary)' }}>
          {isExpanded ? '▼' : '▶'}
        </span>
      </div>
      {isExpanded && (
        <div className="thinking-steps">
          {steps.map((step) => (
            <div key={step.id} className={`thinking-step thinking-step--${step.status}`}>
              <span className="step-icon">
                {step.status === 'active'
                  ? <span className="step-spinner" />
                  : step.status === 'done'
                    ? <span className="step-check">✓</span>
                    : STEP_ICONS[step.id] ?? '○'}
              </span>
              <div className="step-body">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="step-label">{step.label}</span>
                  {step.duration && <span className="step-duration" style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>{step.duration}</span>}
                </div>
                {step.detail && step.status === 'done' && (
                  <pre className="step-detail">{step.detail}</pre>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ChatArea({ currentChat, isLoading, theme, onToggleTheme, onSendMessage, onToggleSidebar, isSidebarCollapsed }) {
  const [inputValue, setInputValue] = useState('');
  const [iterations, setIterations] = useState(1);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentChat]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;
    onSendMessage(inputValue, iterations);
    setInputValue('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const showSteps = currentChat?.steps?.length > 0;
  const showInitialSpinner = currentChat?.isPending && (!currentChat?.steps || currentChat.steps.length === 0);

  return (
    <div className="chat-area">
      <div className="chat-header">
        <button className="toggle-sidebar-btn" onClick={onToggleSidebar}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 12h18M3 6h18M3 18h18" />
          </svg>
        </button>
        <div className="header-title">
          <span className="header-brand">Meta-Scientific Agent</span>
          <span className="header-model">Neuro-Symbolic Discovery System</span>
        </div>
        {/* Theme toggle */}
        <button
          id="theme-toggle-btn"
          className="theme-toggle-btn"
          onClick={onToggleTheme}
          title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {theme === 'dark' ? (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="5"/>
              <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
              <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
          )}
        </button>
      </div>

      <div className="messages-container">
        {!currentChat ? (
          <div className="empty-state">
            <div className="empty-logo">
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                <circle cx="24" cy="24" r="22" stroke="url(#grad)" strokeWidth="2" />
                <path d="M14 24 C14 18 20 14 24 14 C28 14 34 18 34 24 C34 30 28 34 24 34" stroke="url(#grad)" strokeWidth="2" strokeLinecap="round"/>
                <circle cx="24" cy="24" r="4" fill="url(#grad)" />
                <defs>
                  <linearGradient id="grad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#2f81f7"/>
                    <stop offset="1" stopColor="#9b59b6"/>
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <h1>Meta-Scientific Agent</h1>
            <p>Describe the alloy you need. The agent will hypothesize, simulate, and validate autonomously.</p>
            <div className="suggestion-chips">
              {[
                "Design a corrosion-resistant valve alloy",
                "Suggest a high-strength bearing steel",
                "Propose a lightweight aerospace blade alloy",
              ].map(s => (
                <button key={s} className="chip" onClick={() => onSendMessage(s, iterations)}>{s}</button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {/* User Message */}
            <div className="message">
              <div className="avatar user">U</div>
              <div className="message-content user-bubble">
                {currentChat.prompt}
              </div>
            </div>

            {/* AI Response */}
            <div className="message">
              <div className="avatar ai">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <circle cx="10" cy="10" r="9" stroke="white" strokeWidth="1.5"/>
                  <circle cx="10" cy="10" r="3" fill="white"/>
                </svg>
              </div>
              <div className="message-content">
                {showInitialSpinner && (
                  <div className="loading-indicator">
                    <div className="spinner" />
                    <span>Initializing pipeline...</span>
                  </div>
                )}

                {showSteps && (
                  <ThinkingSteps steps={currentChat.steps} isPending={currentChat.isPending} />
                )}

                {!currentChat.isPending && currentChat.markdown && (
                  <>
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        img: ({ node, ...props }) => {
                          const isChart = props.src && !props.src.startsWith('http') && !props.src.startsWith('/');
                          if (isChart) {
                            const themeSuffix = theme === 'dark' ? 'dark' : 'light';
                            const srcWithTheme = props.src.replace('.png', `_${themeSuffix}.png`);
                            return (
                              <figure className="inline-chart-figure">
                                <img
                                  {...props}
                                  src={`${API_BASE_URL}/api/runs/${currentChat.report_id}/${srcWithTheme}`}
                                  loading="lazy"
                                  className="inline-chart-img"
                                />
                              </figure>
                            );
                          }
                          return <img {...props} />;
                        }
                      }}
                    >
                      {currentChat.markdown}
                    </ReactMarkdown>
                  </>
                )}

              </div>
            </div>
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="input-area">
        <form className="input-container" onSubmit={handleSubmit}>
          <textarea
            className="prompt-input"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="E.g., Design a high-strength camshaft alloy..."
            disabled={isLoading}
            rows={1}
          />
          <div className="iterations-wrapper">
            <label className="iterations-label" htmlFor="iterations-select">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="1 4 1 10 7 10" />
                <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
              </svg>
              Retries
            </label>
            <select
              id="iterations-select"
              className="iterations-select"
              value={iterations}
              onChange={(e) => setIterations(Number(e.target.value))}
              disabled={isLoading}
            >
              <option value={1}>1×</option>
              <option value={2}>2×</option>
              <option value={3}>3×</option>
              <option value={5}>5×</option>
            </select>
          </div>
          <button type="submit" className="submit-btn" disabled={!inputValue.trim() || isLoading}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
            </svg>
          </button>
        </form>
        <p className="input-hint">Press Enter to send · Shift+Enter for new line · Retries = auto-adjust on failure</p>
      </div>
    </div>
  );
}

export default ChatArea;
