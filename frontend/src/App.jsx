import React, { useState } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import ProcessingLoader from './components/ProcessingLoader';
import ResultsDashboard from './components/ResultsDashboard';

function App() {
  const [appState, setAppState] = useState('upload'); // upload | processing | results
  const [uploadedFile, setUploadedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('gemini'); // gemini | groq
  
  // Mock data for the MVP flow
  const [mockResults, setMockResults] = useState(null);

  const handleFileUpload = (file) => {
    setUploadedFile(file);
    setAppState('processing');
    
    // Simulate backend processing time (OCR -> LLM Extraction)
    setTimeout(() => {
      setMockResults({
        type: 'prescription',
        items: [
          {
            drugName: 'Amoxicillin 500mg',
            frequency: '1-0-1',
            explanationEn: 'An antibiotic used to treat bacterial infections. Take one pill in the morning and one at night after meals.',
            explanationHi: 'बैक्टीरियल संक्रमण के इलाज के लिए एक एंटीबायोटिक। सुबह और रात के खाने के बाद एक-एक गोली लें।',
            duration: '5 Days',
            instructions: 'Take after meals'
          },
          {
            drugName: 'Paracetamol 650mg',
            frequency: 'SOS',
            explanationEn: 'Used to relieve pain and reduce fever. Take only when you have a fever or body ache.',
            explanationHi: 'दर्द को दूर करने और बुखार कम करने के लिए। केवल तभी लें जब आपको बुखार या बदन दर्द हो।',
            duration: 'As needed',
            instructions: 'Maximum 3 times a day'
          }
        ],
        patient: {
          name: 'Unknown',
          date: new Date().toLocaleDateString()
        }
      });
      setAppState('results');
    }, 4500); // 4.5 seconds of "processing" for effect
  };

  const handleReset = () => {
    setUploadedFile(null);
    setMockResults(null);
    setAppState('upload');
  };

  return (
    <>
      <Header 
        onReset={handleReset} 
        showReset={appState === 'results'} 
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
      />
      
      <main className="container" style={{ padding: 'var(--space-6) var(--space-4)' }}>
        {appState === 'upload' && (
          <div className="animate-fade-in">
            <FileUpload onUpload={handleFileUpload} />
          </div>
        )}
        
        {appState === 'processing' && (
          <div className="animate-fade-in">
            <ProcessingLoader />
          </div>
        )}
        
        {appState === 'results' && mockResults && (
          <div className="animate-fade-in">
            <ResultsDashboard results={mockResults} />
          </div>
        )}
      </main>
    </>
  );
}

export default App;
