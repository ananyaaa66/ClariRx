import React, { useCallback, useState } from 'react';
import { UploadCloud, FileImage, FileText } from 'lucide-react';

const FileUpload = ({ onUpload }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  }, [onUpload]);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'center' }}>
      <h2 style={{ fontSize: 'var(--text-3xl)', marginBottom: 'var(--space-2)' }}>
        Understand your medical reports instantly.
      </h2>
      <p style={{ fontSize: 'var(--text-lg)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-6)' }}>
        Upload a photo of your doctor's handwritten prescription or a lab report to get a clear, plain-language explanation.
      </p>

      <div
        className={`glass-panel glass-panel-hover flex-column flex-center`}
        style={{
          padding: 'var(--space-6)',
          border: isDragging ? '2px dashed var(--color-primary)' : '2px dashed var(--color-card-border)',
          backgroundColor: isDragging ? 'var(--color-primary-light)' : 'var(--color-card-bg)',
          minHeight: '300px',
          cursor: 'pointer',
          transition: 'all 0.3s ease'
        }}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => document.getElementById('fileUpload').click()}
      >
        <input
          type="file"
          id="fileUpload"
          style={{ display: 'none' }}
          onChange={handleChange}
          accept="image/*,.pdf"
        />
        
        <div style={{ 
          backgroundColor: 'var(--color-bg-light)', 
          padding: 'var(--space-4)', 
          borderRadius: 'var(--radius-full)',
          marginBottom: 'var(--space-4)',
          boxShadow: 'var(--shadow-sm)'
        }}>
          <UploadCloud size={48} color="var(--color-primary)" />
        </div>
        
        <h3 style={{ fontSize: 'var(--text-xl)', marginBottom: 'var(--space-2)' }}>
          Click to upload or drag and drop
        </h3>
        <p style={{ color: 'var(--color-text-muted)', marginBottom: 'var(--space-4)' }}>
          Supports JPG, PNG, or PDF files.
        </p>
        
        <div className="flex-center" style={{ gap: 'var(--space-4)' }}>
          <div className="flex-center" style={{ gap: 'var(--space-2)', color: 'var(--color-text-muted)' }}>
            <FileImage size={20} /> <span>Prescriptions</span>
          </div>
          <div className="flex-center" style={{ gap: 'var(--space-2)', color: 'var(--color-text-muted)' }}>
            <FileText size={20} /> <span>Lab Reports</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileUpload;
