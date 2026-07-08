import React, { useState } from 'react';
import ResultCard from './ResultCard';
import { Pill, Clock, Languages } from 'lucide-react';

const ResultsDashboard = ({ results }) => {
  const [lang, setLang] = useState('en'); // 'en' or 'hi'

  const toggleLang = () => setLang(prev => prev === 'en' ? 'hi' : 'en');

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto' }}>
      
      <div className="flex-between" style={{ marginBottom: 'var(--space-4)' }}>
        <div>
          <h2 style={{ fontSize: 'var(--text-3xl)', marginBottom: 'var(--space-1)' }}>Prescription Details</h2>
          <p style={{ color: 'var(--color-text-muted)', fontSize: 'var(--text-lg)' }}>
            Found {results.items.length} medicines
          </p>
        </div>
        
        <button 
          onClick={toggleLang} 
          className="btn btn-outline"
          style={{ backgroundColor: 'var(--color-card-bg)', backdropFilter: 'blur(10px)' }}
        >
          <Languages size={20} />
          <span>{lang === 'en' ? 'हिंदी में पढ़ें' : 'Read in English'}</span>
        </button>
      </div>

      <div className="flex-column" style={{ gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
        {results.items.map((item, idx) => (
          <ResultCard key={idx} item={item} lang={lang} />
        ))}
      </div>

      {/* Reminder Call to Action */}
      <div className="glass-panel flex-between" style={{ padding: 'var(--space-4)', backgroundColor: 'var(--color-primary-light)' }}>
        <div className="flex-center" style={{ gap: 'var(--space-3)' }}>
          <div style={{ padding: 'var(--space-2)', backgroundColor: 'var(--color-primary)', borderRadius: 'var(--radius-full)', color: 'white' }}>
            <Clock size={24} />
          </div>
          <div>
            <h3 style={{ fontSize: 'var(--text-xl)' }}>Set up Reminders</h3>
            <p style={{ color: 'var(--color-text-muted)' }}>Get notified when it's time to take your medicines.</p>
          </div>
        </div>
        <button className="btn btn-primary">
          Enable Reminders
        </button>
      </div>

    </div>
  );
};

export default ResultsDashboard;
