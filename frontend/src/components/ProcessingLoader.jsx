import React, { useState, useEffect } from 'react';
import { ScanFace, Activity, CheckCircle2 } from 'lucide-react';

const ProcessingLoader = () => {
  const [step, setStep] = useState(0);
  
  const steps = [
    { icon: <ScanFace size={32} />, text: "Scanning document..." },
    { icon: <Activity size={32} />, text: "Extracting medical terminology..." },
    { icon: <CheckCircle2 size={32} />, text: "Simplifying explanations..." }
  ];

  useEffect(() => {
    const timer1 = setTimeout(() => setStep(1), 1500);
    const timer2 = setTimeout(() => setStep(2), 3000);
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  return (
    <div className="flex-column flex-center" style={{ minHeight: '50vh', gap: 'var(--space-6)' }}>
      
      <div 
        className="flex-center animate-pulse-ring" 
        style={{ 
          width: '120px', 
          height: '120px', 
          borderRadius: 'var(--radius-full)',
          backgroundColor: 'var(--color-primary-light)',
          color: 'var(--color-primary)'
        }}
      >
        {steps[step].icon}
      </div>

      <div style={{ textAlign: 'center' }}>
        <h2 style={{ fontSize: 'var(--text-2xl)', marginBottom: 'var(--space-2)' }}>
          Processing your report
        </h2>
        <p className="animate-fade-in" key={step} style={{ fontSize: 'var(--text-lg)', color: 'var(--color-primary)' }}>
          {steps[step].text}
        </p>
      </div>
      
    </div>
  );
};

export default ProcessingLoader;
