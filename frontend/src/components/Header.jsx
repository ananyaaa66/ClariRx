import React from 'react';
import { Stethoscope, ArrowLeft } from 'lucide-react';

const Header = ({ onReset, showReset }) => {
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
    </header>
  );
};

export default Header;
