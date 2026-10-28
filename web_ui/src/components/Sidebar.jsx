import React, { useState } from 'react';

function Sidebar({ isCollapsed, history, onNewChat, onSelectHistory, onDeleteHistory }) {
  const [deleteTarget, setDeleteTarget] = useState(null);

  const handleDeleteClick = (e, item) => {
    e.stopPropagation();
    setDeleteTarget(item);
  };

  const confirmDelete = () => {
    if (deleteTarget) {
      onDeleteHistory(deleteTarget.report_id);
      setDeleteTarget(null);
    }
  };

  const cancelDelete = () => {
    setDeleteTarget(null);
  };

  return (
    <div className={`sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <div className="sidebar-title">Meta-Scientific</div>
        <button className="new-chat-btn" onClick={onNewChat} title="New Chat">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 5v14M5 12h14" />
          </svg>
        </button>
      </div>
      <div className="history-list">
        {history.length === 0 ? (
          <div style={{ padding: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem', textAlign: 'center' }}>
            No previous reports
          </div>
        ) : (
          history.map((item) => (
            <div 
              key={item.report_id} 
              className="history-item"
              onClick={() => onSelectHistory(item)}
              title={item.goal}
            >
              <span className="history-item-text">{item.goal}</span>
              <button
                className="history-delete-btn"
                onClick={(e) => handleDeleteClick(e, item)}
                title="Delete report"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="delete-modal-overlay" onClick={cancelDelete}>
          <div className="delete-modal" onClick={(e) => e.stopPropagation()}>
            <div className="delete-modal-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
                <line x1="10" y1="11" x2="10" y2="17" />
                <line x1="14" y1="11" x2="14" y2="17" />
              </svg>
            </div>
            <h3 className="delete-modal-title">Delete Report</h3>
            <p className="delete-modal-text">
              This query report will be permanently deleted along with all its charts and data. This action cannot be undone.
            </p>
            <div className="delete-modal-actions">
              <button className="delete-modal-cancel" onClick={cancelDelete}>Cancel</button>
              <button className="delete-modal-confirm" onClick={confirmDelete}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Sidebar;
