import React from 'react';
import { Pill, AlertCircle, Info } from 'lucide-react';

const ResultCard = ({ item, lang }) => {
  return (
    <div className="glass-panel glass-panel-hover" style={{ padding: 'var(--space-4)' }}>
      
      <div className="flex-between" style={{ borderBottom: '1px solid var(--color-card-border)', paddingBottom: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
        <div className="flex-center" style={{ gap: 'var(--space-3)' }}>
          <div style={{ padding: '12px', backgroundColor: 'var(--color-bg-light)', borderRadius: 'var(--radius-md)' }}>
            <Pill size={28} color="var(--color-primary)" />
          </div>
          <div>
            <h3 style={{ fontSize: 'var(--text-2xl)', color: 'var(--color-primary)' }}>{item.drugName}</h3>
            <div className="flex-center" style={{ gap: 'var(--space-2)', justifyContent: 'flex-start' }}>
              <span style={{ 
                backgroundColor: 'var(--color-primary-light)', 
                color: 'var(--color-primary-hover)', 
                padding: '4px 12px', 
                borderRadius: 'var(--radius-full)',
                fontSize: 'var(--text-sm)',
                fontWeight: '600'
              }}>
                {item.frequency}
              </span>
              <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>
                • {item.duration}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div style={{ backgroundColor: 'var(--color-bg-light)', padding: 'var(--space-3)', borderRadius: 'var(--radius-md)' }}>
        <div className="flex-center" style={{ gap: 'var(--space-2)', marginBottom: 'var(--space-2)', alignItems: 'flex-start' }}>
          <Info size={20} color="var(--color-accent)" style={{ flexShrink: 0, marginTop: '2px' }} />
          <p style={{ fontSize: 'var(--text-lg)', lineHeight: '1.5' }}>
            {lang === 'en' ? item.explanationEn : item.explanationHi}
          </p>
        </div>
        
        {item.instructions && (
          <div className="flex-center" style={{ gap: 'var(--space-2)', marginTop: 'var(--space-3)', alignItems: 'flex-start' }}>
            <AlertCircle size={20} color="var(--color-warning)" style={{ flexShrink: 0, marginTop: '2px' }} />
            <p style={{ fontSize: 'var(--text-base)', color: 'var(--color-warning)' }}>
              <strong>{lang === 'en' ? 'Important:' : 'महत्वपूर्ण:'}</strong> {item.instructions}
            </p>
          </div>
        )}
      </div>
      
    </div>
  );
};

export default ResultCard;
