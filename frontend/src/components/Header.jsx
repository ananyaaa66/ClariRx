import React from 'react';
import { Stethoscope, ArrowLeft, BrainCircuit } from 'lucide-react';

const Header = ({ onReset, showReset, selectedModel, onModelChange }) => {
  return (
    <header className="glass-panel" style={{ 
      margin: 'var(--space-3)',
      padding: 'var(--space-3) var(--space-4)',
      position: 'sticky',
      top: 'var(--space-3)',
      zIndex: 100
    }}>
      <div className="flex-between">
        <div className="flex-center" style={{ gap: 'var(--space-2)' }}>
          <div style={{ 
            backgroundColor: 'var(--color-primary)', 
            padding: '8px', 
            borderRadius: 'var(--radius-md)',
            color: 'white'
          }}>
            <Stethoscope size={24} />
          </div>
          <div>
            <h1 style={{ fontSize: 'var(--text-xl)', color: 'var(--color-primary)' }}>ClariRx</h1>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>Clarity in every prescription</p>
          </div>
        </div>
        
        <div className="flex-center" style={{ gap: 'var(--space-3)' }}>
          {/* Model Selector Dropdown */}
          <div className="flex-center" style={{ 
            backgroundColor: 'var(--color-bg-light)', 
            padding: '4px 12px', 
            borderRadius: 'var(--radius-full)',
            border: '1px solid var(--color-card-border)'
          }}>
            <BrainCircuit size={16} color="var(--color-primary)" style={{ marginRight: '8px' }} />
            <select 
              value={selectedModel}
              onChange={(e) => onModelChange(e.target.value)}
              style={{
                border: 'none',
                backgroundColor: 'transparent',
                fontFamily: 'var(--font-body)',
                fontSize: 'var(--text-sm)',
                color: 'var(--color-text-main)',
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              <option value="gemini">Gemini 2.5 Flash</option>
              <option value="groq">Groq (Llama 3)</option>
            </select>
          </div>

          {showReset && (
            <button 
              onClick={onReset}
              className="btn btn-outline"
              style={{ padding: 'var(--space-2) var(--space-3)' }}
            >
              <ArrowLeft size={18} />
              <span>Upload Another</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
