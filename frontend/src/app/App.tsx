import { useState, useRef, useCallback } from "react";
import {
  Stethoscope,
  Upload,
  Bell,
  History,
  ArrowRight,
  Languages,
  Clock,
  Phone,
  Download,
  CheckCircle,
  Loader2,
  Plus,
  Trash2,
  ChevronDown,
  AlertTriangle,
  FileImage,
  FileText,
  ChevronRight,
  Calendar,
  Pill,
} from "lucide-react";

type Page = "home" | "upload" | "results" | "reminders" | "history";
type Language = "en" | "hi";
type Model = "gemini-2.5-flash" | "groq-llama3";

interface Medicine {
  id: string;
  drugName: string;
  drugNameHi: string;
  frequency: string;
  duration: string;
  explanationEn: string;
  explanationHi: string;
  instructions: string;
  instructionsHi: string;
}

interface Patient {
  name: string;
  age: string;
  date: string;
  doctor: string;
}

interface Reminder {
  id: string;
  drugName: string;
  time: string;
  phone: string;
  frequency: string;
}

interface HistoryEntry {
  id: string;
  date: string;
  patient: string;
  medicines: number;
  type: "Prescription" | "Lab Report";
}

const mockPatient: Patient = {
  name: "Ramesh Kumar",
  age: "67 years",
  date: "22 July 2025",
  doctor: "Dr. Priya Sharma, MBBS, MD",
};

const mockMedicines: Medicine[] = [
  {
    id: "1",
    drugName: "Metformin 500mg",
    drugNameHi: "मेटफॉर्मिन 500mg",
    frequency: "1-0-1",
    duration: "30 Days",
    explanationEn:
      "Controls blood sugar by reducing glucose release from the liver and improving the body's response to insulin. One of the most widely prescribed and safest medicines for Type 2 diabetes worldwide.",
    explanationHi:
      "लीवर से ग्लूकोज़ के स्राव को कम करके और इंसुलिन के प्रति शरीर की प्रतिक्रिया को बेहतर बनाकर रक्त शर्करा को नियंत्रित करती है। टाइप 2 मधुमेह के लिए सबसे सुरक्षित दवाओं में से एक।",
    instructions: "Take after meals. Do not crush or chew the tablet.",
    instructionsHi: "भोजन के बाद लें। गोली को कुचलें या चबाएं नहीं।",
  },
  {
    id: "2",
    drugName: "Amlodipine 5mg",
    drugNameHi: "एम्लोडिपिन 5mg",
    frequency: "0-0-1",
    duration: "30 Days",
    explanationEn:
      "Prescribed for high blood pressure. Relaxes and widens blood vessels so the heart doesn't have to work as hard. Helps reduce the risk of heart attack and stroke over time.",
    explanationHi:
      "उच्च रक्तचाप के लिए निर्धारित। रक्त वाहिकाओं को शिथिल और चौड़ा करता है ताकि हृदय को कम मेहनत करनी पड़े। समय के साथ हार्ट अटैक और स्ट्रोक का जोखिम कम करता है।",
    instructions:
      "Take at bedtime. Never stop suddenly without consulting your doctor.",
    instructionsHi:
      "रात को सोने से पहले लें। बिना डॉक्टर की सलाह के अचानक बंद न करें।",
  },
  {
    id: "3",
    drugName: "Pantoprazole 40mg",
    drugNameHi: "पैंटोप्राज़ोल 40mg",
    frequency: "1-0-0",
    duration: "14 Days",
    explanationEn:
      "Reduces stomach acid production. Prescribed here to protect the stomach lining and prevent acidity or irritation caused by the other medicines in this prescription.",
    explanationHi:
      "पेट में एसिड के उत्पादन को कम करता है। इस प्रिस्क्रिप्शन की अन्य दवाओं से होने वाली एसिडिटी और जलन से बचाव के लिए निर्धारित।",
    instructions: "Take 30 minutes before breakfast on an empty stomach.",
    instructionsHi: "नाश्ते से 30 मिनट पहले खाली पेट लें।",
  },
];

const mockHistory: HistoryEntry[] = [
  { id: "1", date: "22 Jul 2025", patient: "Ramesh Kumar", medicines: 3, type: "Prescription" },
  { id: "2", date: "14 Jun 2025", patient: "Ramesh Kumar", medicines: 2, type: "Lab Report" },
  { id: "3", date: "02 May 2025", patient: "Sunita Kumar", medicines: 4, type: "Prescription" },
  { id: "4", date: "18 Mar 2025", patient: "Ramesh Kumar", medicines: 5, type: "Prescription" },
  { id: "5", date: "05 Jan 2025", patient: "Sunita Kumar", medicines: 2, type: "Lab Report" },
];

// ─── API Helper Functions ───────────────────────────────────────────────────────

const API_BASE = "http://localhost:8000";

async function apiUploadDocument(file: File, docType = "prescription") {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("doc_type", docType);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.statusText}`);
  }
  return await res.json();
}

async function apiExplainMedicine(drugName: string, frequency = "", duration = "", instructions = "") {
  const res = await fetch(`${API_BASE}/api/explain/medicine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      drug_name: drugName,
      frequency,
      duration,
      instructions,
    }),
  });
  if (!res.ok) {
    throw new Error(`Explanation failed: ${res.statusText}`);
  }
  return await res.json();
}

// ─── Root ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [page, setPage] = useState<Page>("home");
  const [selectedModel, setSelectedModel] = useState<Model>("gemini-2.5-flash");
  const [language, setLanguage] = useState<Language>("en");
  const [processingStage, setProcessingStage] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [patient, setPatient] = useState<Patient>(mockPatient);
  const [medicines, setMedicines] = useState<Medicine[]>(mockMedicines);
  const [reminders, setReminders] = useState<Reminder[]>([
    { id: "1", drugName: "Metformin 500mg", time: "08:00", phone: "+91 98765 43210", frequency: "Twice daily" },
    { id: "2", drugName: "Amlodipine 5mg", time: "21:00", phone: "+91 98765 43210", frequency: "Once daily" },
  ]);
  const [form, setForm] = useState({ drugName: "", time: "", phone: "", frequency: "Once daily" });

  const navigate = (p: Page) => {
    setPage(p);
    window.scrollTo({ top: 0 });
  };

  const startProcessing = async (file?: File) => {
    setIsProcessing(true);
    setProcessingStage(1);

    if (file) {
      try {
        // 1. Send file to backend FastAPI server
        const uploadData = await apiUploadDocument(file, "prescription");
        setProcessingStage(2);

        const rawItems = uploadData.items || [];
        const resultMeds: Medicine[] = [];

        // 2. Obtain grounded English & Hindi explanations for extracted entities
        for (let i = 0; i < rawItems.length; i++) {
          const item = rawItems[i];
          try {
            const exp = await apiExplainMedicine(
              item.drug_name || "Unknown Medicine",
              item.frequency || "",
              item.duration || "",
              item.instructions || ""
            );

            resultMeds.push({
              id: (i + 1).toString(),
              drugName: exp.brand_name ? `${exp.brand_name} (${exp.generic_name})` : (item.drug_name || exp.generic_name || "Medicine"),
              drugNameHi: exp.generic_name || item.drug_name || "दवा",
              frequency: item.frequency || "1-0-1",
              duration: item.duration || "As directed",
              explanationEn: exp.plain_english_use || "Prescribed medicine for your treatment.",
              explanationHi: exp.plain_hindi_use || "आपके इलाज के लिए निर्धारित दवा।",
              instructions: exp.plain_english_instructions || item.instructions || "Take as instructed by doctor.",
              instructionsHi: exp.plain_hindi_instructions || "डॉक्टर के निर्देशानुसार लें।",
            });
          } catch {
            resultMeds.push({
              id: (i + 1).toString(),
              drugName: item.drug_name || "Medicine",
              drugNameHi: item.drug_name || "दवा",
              frequency: item.frequency || "1-0-1",
              duration: item.duration || "5 Days",
              explanationEn: "Prescribed medicine for your treatment.",
              explanationHi: "आपके इलाज के लिए निर्धारित दवा।",
              instructions: item.instructions || "Take as directed by doctor.",
              instructionsHi: "डॉक्टर के निर्देशानुसार लें।",
            });
          }
        }

        setProcessingStage(3);
        await new Promise((r) => setTimeout(r, 600));

        if (resultMeds.length > 0) {
          setMedicines(resultMeds);
        } else {
          setMedicines(mockMedicines);
        }

        setPatient({
          name: file.name.replace(/\.[^/.]+$/, "") || "Patient",
          age: "Uploaded File",
          date: new Date().toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }),
          doctor: "ClariRx AI Pipeline",
        });
      } catch (err) {
        console.warn("Backend server offline or endpoint error. Using fallback simulation:", err);
        setProcessingStage(2);
        await new Promise((r) => setTimeout(r, 1500));
        setProcessingStage(3);
        await new Promise((r) => setTimeout(r, 1500));
        setMedicines(mockMedicines);
      } finally {
        setIsProcessing(false);
        setProcessingStage(0);
        navigate("results");
      }
    } else {
      // Fallback for simulated trigger without file
      setTimeout(() => setProcessingStage(2), 1500);
      setTimeout(() => setProcessingStage(3), 3000);
      setTimeout(() => {
        setIsProcessing(false);
        setProcessingStage(0);
        navigate("results");
      }, 4200);
    }
  };

  const addReminder = () => {
    if (!form.drugName || !form.time || !form.phone) return;
    setReminders((p) => [...p, { id: Date.now().toString(), ...form }]);
    setForm({ drugName: "", time: "", phone: "", frequency: "Once daily" });
  };

  return (
    <div className="min-h-screen bg-[#FBFBF9]" style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }}>
      <Navbar currentPage={page} navigate={navigate} selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
      {page === "home"      && <HomePage navigate={navigate} />}
      {page === "upload"    && <UploadPage isDragging={isDragging} setIsDragging={setIsDragging} isProcessing={isProcessing} processingStage={processingStage} startProcessing={startProcessing} />}
      {page === "results"   && <ResultsPage language={language} setLanguage={setLanguage} patient={patient} medicines={medicines} navigate={navigate} />}
      {page === "reminders" && <RemindersPage reminders={reminders} form={form} setForm={setForm} addReminder={addReminder} removeReminder={id => setReminders(p => p.filter(r => r.id !== id))} />}
      {page === "history"   && <HistoryPage history={mockHistory} navigate={navigate} />}
    </div>
  );
}

// ─── Navbar ───────────────────────────────────────────────────────────────────

function Navbar({ currentPage, navigate, selectedModel, setSelectedModel }: {
  currentPage: Page;
  navigate: (p: Page) => void;
  selectedModel: Model;
  setSelectedModel: (m: Model) => void;
}) {
  const links: { page: Page; label: string }[] = [
    { page: "home",      label: "Home" },
    { page: "upload",    label: "Upload" },
    { page: "reminders", label: "Reminders" },
    { page: "history",   label: "Saved Reports" },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#FBFBF9]/95 backdrop-blur-sm border-b border-[#E5E7EB]">
      <div className="max-w-5xl mx-auto px-6 h-14 flex items-center gap-8">
        {/* Logo */}
        <button onClick={() => navigate("home")} className="flex items-center gap-2 flex-shrink-0 mr-2">
          <Stethoscope className="w-[18px] h-[18px] text-[#0D9488]" strokeWidth={2} />
          <span className="text-[#111827] font-bold text-[16px] tracking-tight">ClariRx</span>
        </button>

        {/* Nav links */}
        <nav className="hidden md:flex items-center gap-0">
          {links.map(({ page, label }) => (
            <button
              key={page}
              onClick={() => navigate(page)}
              className={`relative px-3.5 py-4 text-[13.5px] font-semibold transition-colors ${
                currentPage === page ? "text-[#0D9488]" : "text-[#6B7280] hover:text-[#111827]"
              }`}
            >
              {label}
              {currentPage === page && (
                <span className="absolute bottom-0 left-3.5 right-3.5 h-[2px] bg-[#0D9488] rounded-full" />
              )}
            </button>
          ))}
        </nav>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Tagline */}
        <span className="hidden lg:block text-[11px] text-[#9CA3AF] tracking-wide">
          Clarity in every prescription
        </span>

        {/* Model switcher */}
        <div className="relative">
          <select
            value={selectedModel}
            onChange={e => setSelectedModel(e.target.value as Model)}
            className="appearance-none bg-transparent text-[#6B7280] text-[12px] font-semibold pr-5 pl-0 py-1 focus:outline-none cursor-pointer hover:text-[#111827] transition-colors"
          >
            <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
            <option value="groq-llama3">Groq Llama 3</option>
          </select>
          <ChevronDown className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 text-[#9CA3AF] pointer-events-none" />
        </div>
      </div>

      {/* Mobile nav */}
      <div className="md:hidden flex gap-0 px-4 border-t border-[#E5E7EB] overflow-x-auto">
        {links.map(({ page, label }) => (
          <button
            key={page}
            onClick={() => navigate(page)}
            className={`relative px-3 py-2.5 text-[12px] font-semibold whitespace-nowrap transition-colors ${
              currentPage === page ? "text-[#0D9488]" : "text-[#9CA3AF]"
            }`}
          >
            {label}
            {currentPage === page && <span className="absolute bottom-0 left-2 right-2 h-[2px] bg-[#0D9488] rounded-full" />}
          </button>
        ))}
      </div>
    </header>
  );
}

// ─── Home ─────────────────────────────────────────────────────────────────────

function HomePage({ navigate }: { navigate: (p: Page) => void }) {
  return (
    <div className="max-w-5xl mx-auto px-6">

      {/* Hero */}
      <section className="pt-20 pb-16 max-w-[640px]">
        <p className="text-[11px] font-bold text-[#0D9488] tracking-[0.15em] uppercase mb-6">
          100% Private &amp; Secure
        </p>
        <h1 className="text-[3rem] md:text-[3.6rem] font-extrabold text-[#111827] leading-[1.08] tracking-tight mb-6">
          Your prescription,<br />in plain language.
        </h1>
        <p className="text-[17px] text-[#6B7280] leading-[1.7] mb-10 max-w-[480px]">
          ClariRx helps elderly patients and families understand prescriptions
          and lab reports — explained clearly in English and Hindi, without
          medical jargon.
        </p>
        <div className="flex items-center gap-5">
          <button
            onClick={() => navigate("upload")}
            className="inline-flex items-center gap-2 bg-[#0D9488] text-white font-bold text-[14.5px] px-6 py-3.5 rounded-xl hover:bg-[#0b8277] active:scale-[0.98] transition-all"
          >
            Analyze Prescription
            <ArrowRight className="w-4 h-4" />
          </button>
          <button
            onClick={() => navigate("history")}
            className="text-[14px] text-[#6B7280] font-semibold hover:text-[#111827] transition-colors"
          >
            View saved reports →
          </button>
        </div>
      </section>

      {/* Divider */}
      <div className="border-t border-[#E5E7EB]" />

      {/* How it works — editorial list */}
      <section className="py-14">
        <p className="text-[11px] font-bold text-[#9CA3AF] tracking-[0.15em] uppercase mb-10">
          How it works
        </p>
        <div className="grid md:grid-cols-3 gap-0 md:divide-x divide-[#E5E7EB]">
          {[
            {
              n: "01",
              title: "Upload",
              desc: "Take a photo or upload a PDF of any handwritten or printed prescription.",
            },
            {
              n: "02",
              title: "Auto Extract",
              desc: "OCR and BioBERT models extract medicine names, dosages, frequency, and instructions.",
            },
            {
              n: "03",
              title: "Read clearly",
              desc: "Every medicine explained in plain English or Hindi — no jargon, no confusion.",
            },
          ].map((s, i) => (
            <div key={i} className="px-0 md:px-8 first:pl-0 py-4 md:py-0">
              <span className="text-[11px] font-bold text-[#D1D5DB] tracking-widest block mb-4">
                {s.n}
              </span>
              <h3 className="text-[18px] font-extrabold text-[#111827] mb-2">{s.title}</h3>
              <p className="text-[14px] text-[#6B7280] leading-[1.65]">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

// ─── Upload ───────────────────────────────────────────────────────────────────

function UploadPage({ isDragging, setIsDragging, isProcessing, processingStage, startProcessing }: {
  isDragging: boolean;
  setIsDragging: (v: boolean) => void;
  isProcessing: boolean;
  processingStage: number;
  startProcessing: (file?: File) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      startProcessing(e.dataTransfer.files[0]);
    } else {
      startProcessing();
    }
  }, [startProcessing, setIsDragging]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      startProcessing(e.target.files[0]);
    } else {
      startProcessing();
    }
  };

  const stages = [
    { label: "Running OCR", sub: "Extracting text from image or PDF" },
    { label: "BioBERT Entity Lookup", sub: "Identifying medicines, dosages & instructions" },
    { label: "Generating Hindi Translation", sub: "Translating explanations to plain Hindi" },
  ];

  return (
    <div className="max-w-lg mx-auto px-6 py-14">
      <h1 className="text-[2rem] font-extrabold text-[#111827] mb-1.5">Upload Prescription</h1>
      <p className="text-[15px] text-[#6B7280] mb-10">Supports JPG, PNG, and PDF — up to 10 MB.</p>

      {/* Dropzone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onClick={() => fileRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl flex flex-col items-center justify-center py-16 px-8 cursor-pointer transition-all duration-150 ${
          isDragging ? "border-[#0D9488] bg-[#E6F4F1]/40" : "border-[#D1D5DB] hover:border-[#9CA3AF]"
        }`}
      >
        <input ref={fileRef} type="file" accept=".jpg,.jpeg,.png,.pdf" className="hidden" onChange={handleFileChange} />
        <div className="flex items-center gap-3 mb-5 opacity-40">
          <FileImage className="w-8 h-8 text-[#111827]" strokeWidth={1.5} />
          <FileText className="w-8 h-8 text-[#111827]" strokeWidth={1.5} />
        </div>
        <p className="text-[16px] font-bold text-[#111827] mb-1">
          {isDragging ? "Drop to upload" : "Drag & drop here"}
        </p>
        <p className="text-[13px] text-[#9CA3AF]">or click to browse — JPG, PNG, PDF</p>
      </div>

      {/* Tip */}
      <p className="mt-5 text-[13px] text-[#9CA3AF] leading-snug">
        Tip: ensure the prescription is well-lit and in focus for best OCR accuracy.
      </p>

      {/* Processing overlay */}
      {isProcessing && (
        <div className="fixed inset-0 bg-[#FBFBF9]/90 backdrop-blur-sm z-50 flex items-center justify-center p-6">
          <div className="bg-white border border-[#E5E7EB] rounded-2xl p-8 w-full max-w-sm">
            <p className="text-[11px] font-bold text-[#9CA3AF] tracking-[0.15em] uppercase mb-1">
              Processing
            </p>
            <h3 className="text-[20px] font-extrabold text-[#111827] mb-7">
              Analyzing prescription…
            </h3>

            <div className="space-y-5">
              {stages.map((stage, i) => {
                const n = i + 1;
                const done = processingStage > n;
                const active = processingStage === n;
                const idle = processingStage < n;
                return (
                  <div key={i} className="flex items-start gap-4">
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 transition-all ${
                      done ? "bg-[#0D9488]" : active ? "bg-transparent" : "bg-transparent"
                    }`}>
                      {done && <CheckCircle className="w-5 h-5 text-[#0D9488]" />}
                      {active && <Loader2 className="w-4 h-4 text-[#0D9488] animate-spin" />}
                      {idle && <span className="w-4 h-4 rounded-full border-2 border-[#E5E7EB] block" />}
                    </div>
                    <div className={idle ? "opacity-40" : ""}>
                      <p className={`text-[14px] font-bold ${done || active ? "text-[#111827]" : "text-[#9CA3AF]"}`}>
                        {stage.label}
                      </p>
                      <p className="text-[12px] text-[#9CA3AF] mt-0.5">{stage.sub}</p>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Thin progress line */}
            <div className="mt-8 h-px bg-[#E5E7EB] overflow-hidden rounded-full">
              <div
                className="h-px bg-[#0D9488] transition-all duration-700 ease-out"
                style={{ width: `${(processingStage / 3) * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Results ──────────────────────────────────────────────────────────────────

function ResultsPage({ language, setLanguage, patient, medicines, navigate }: {
  language: Language;
  setLanguage: (l: Language) => void;
  patient: Patient;
  medicines: Medicine[];
  navigate: (p: Page) => void;
}) {
  const hi = language === "hi";

  return (
    <div className="max-w-[660px] mx-auto px-6 py-10">
      {/* Sticky header */}
      <div className="sticky top-14 z-30 bg-[#FBFBF9]/95 backdrop-blur-sm pt-4 pb-4 -mx-6 px-6 border-b border-[#E5E7EB] mb-8">
        <div className="flex items-center justify-between">
          <p className="text-[11px] font-bold text-[#9CA3AF] tracking-[0.15em] uppercase">
            {hi ? "प्रिस्क्रिप्शन विवरण" : "Prescription Details"}
          </p>
          <button
            onClick={() => setLanguage(hi ? "en" : "hi")}
            className="flex items-center gap-1.5 text-[12.5px] font-bold text-[#0D9488] hover:text-[#0b8277] transition-colors"
          >
            <Languages className="w-3.5 h-3.5" />
            {hi ? "Read in English" : "हिंदी में पढ़ें"}
          </button>
        </div>
      </div>

      {/* Patient strip */}
      <div className="flex flex-wrap gap-x-8 gap-y-3 pb-7 border-b border-[#E5E7EB] mb-8">
        {[
          { label: hi ? "मरीज़" : "Patient",   value: patient.name },
          { label: hi ? "आयु" : "Age",         value: patient.age },
          { label: hi ? "तारीख" : "Date",      value: patient.date },
          { label: hi ? "डॉक्टर" : "Doctor",   value: patient.doctor },
        ].map((item, i) => (
          <div key={i}>
            <p className="text-[10px] font-bold text-[#9CA3AF] tracking-[0.12em] uppercase mb-0.5">
              {item.label}
            </p>
            <p className="text-[14px] font-semibold text-[#111827]">{item.value}</p>
          </div>
        ))}
      </div>

      {/* Medicines */}
      <div className="space-y-0 divide-y divide-[#E5E7EB]">
        {medicines.map(med => (
          <MedicineRow key={med.id} medicine={med} language={language} />
        ))}
      </div>

      {/* Reminders CTA */}
      <div className="mt-12 pt-7 border-t border-[#E5E7EB] flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <p className="text-[15px] font-extrabold text-[#111827] mb-0.5">
            {hi ? "दवा के रिमाइंडर सेट करें" : "Set up medicine reminders"}
          </p>
          <p className="text-[13px] text-[#9CA3AF]">
            {hi ? "SMS या WhatsApp पर सही समय पर याद दिलाएं।" : "Get SMS or WhatsApp alerts so no dose is ever missed."}
          </p>
        </div>
        <button
          onClick={() => navigate("reminders")}
          className="flex items-center gap-2 bg-[#0D9488] text-white font-bold px-5 py-3 rounded-xl text-[13.5px] hover:bg-[#0b8277] transition-colors whitespace-nowrap"
        >
          <Bell className="w-4 h-4" />
          {hi ? "रिमाइंडर" : "Set up Reminders"}
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}

function MedicineRow({ medicine, language }: { medicine: Medicine; language: Language }) {
  const hi = language === "hi";

  return (
    <div className="py-8">
      {/* Drug name + badges */}
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2 mb-4">
        <h2 className="text-[1.55rem] font-extrabold text-[#111827] leading-tight">
          {hi ? medicine.drugNameHi : medicine.drugName}
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-[12px] font-bold text-[#0D9488] border border-[#0D9488]/30 bg-[#E6F4F1] px-2.5 py-0.5 rounded-md tracking-wide">
            {medicine.frequency}
          </span>
          <span className="text-[11px] text-[#9CA3AF] font-semibold">
            {hi ? "सुबह–दोपहर–रात" : "Morning–Afternoon–Night"}
          </span>
          <span className="text-[12px] text-[#6B7280] font-semibold border border-[#E5E7EB] px-2.5 py-0.5 rounded-md">
            {medicine.duration}
          </span>
        </div>
      </div>

      {/* Explanation */}
      <p className="text-[15.5px] text-[#374151] leading-[1.72] mb-4 max-w-[560px]">
        {hi ? medicine.explanationHi : medicine.explanationEn}
      </p>

      {/* Instruction */}
      <div className="flex items-start gap-2">
        <AlertTriangle className="w-3.5 h-3.5 text-[#D97706] flex-shrink-0 mt-[3px]" />
        <p className="text-[13.5px] font-semibold text-[#92400E] leading-snug">
          {hi ? medicine.instructionsHi : medicine.instructions}
        </p>
      </div>
    </div>
  );
}

// ─── Reminders ────────────────────────────────────────────────────────────────

function RemindersPage({ reminders, form, setForm, addReminder, removeReminder }: {
  reminders: Reminder[];
  form: { drugName: string; time: string; phone: string; frequency: string };
  setForm: (f: typeof form) => void;
  addReminder: () => void;
  removeReminder: (id: string) => void;
}) {
  const inputCls = "w-full bg-white border border-[#E5E7EB] rounded-xl px-4 py-3 text-[15px] text-[#111827] placeholder:text-[#C4C9D4] focus:outline-none focus:border-[#0D9488] transition-colors";

  return (
    <div className="max-w-[580px] mx-auto px-6 py-12">
      <h1 className="text-[2rem] font-extrabold text-[#111827] mb-1.5">Medicine Reminders</h1>
      <p className="text-[15px] text-[#6B7280] mb-10">Schedule SMS or WhatsApp reminders so no dose is missed.</p>

      {/* Form */}
      <div className="space-y-4 mb-4">
        <div>
          <label className="block text-[11px] font-bold text-[#9CA3AF] tracking-[0.12em] uppercase mb-2">
            Medicine Name
          </label>
          <input
            type="text"
            placeholder="e.g. Metformin 500mg"
            value={form.drugName}
            onChange={e => setForm({ ...form, drugName: e.target.value })}
            className={inputCls}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-[11px] font-bold text-[#9CA3AF] tracking-[0.12em] uppercase mb-2">
              Reminder Time
            </label>
            <input
              type="time"
              value={form.time}
              onChange={e => setForm({ ...form, time: e.target.value })}
              className={inputCls}
            />
          </div>
          <div>
            <label className="block text-[11px] font-bold text-[#9CA3AF] tracking-[0.12em] uppercase mb-2">
              Frequency
            </label>
            <select
              value={form.frequency}
              onChange={e => setForm({ ...form, frequency: e.target.value })}
              className={inputCls + " appearance-none cursor-pointer"}
            >
              <option>Once daily</option>
              <option>Twice daily</option>
              <option>Three times daily</option>
              <option>Every alternate day</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-[11px] font-bold text-[#9CA3AF] tracking-[0.12em] uppercase mb-2">
            Phone Number (SMS / WhatsApp)
          </label>
          <input
            type="tel"
            placeholder="+91 98765 43210"
            value={form.phone}
            onChange={e => setForm({ ...form, phone: e.target.value })}
            className={inputCls}
          />
        </div>
      </div>

      <button
        onClick={addReminder}
        className="flex items-center gap-2 bg-[#0D9488] text-white font-bold px-5 py-3 rounded-xl text-[14px] hover:bg-[#0b8277] active:scale-[0.98] transition-all mb-12"
      >
        <Plus className="w-4 h-4" />
        Schedule Reminder
      </button>

      {/* Divider + list */}
      <div className="border-t border-[#E5E7EB] pt-8">
        <p className="text-[11px] font-bold text-[#9CA3AF] tracking-[0.15em] uppercase mb-5">
          Active reminders &mdash; {reminders.length}
        </p>

        {reminders.length === 0 ? (
          <p className="text-[14px] text-[#C4C9D4] py-6">No reminders scheduled yet.</p>
        ) : (
          <div className="divide-y divide-[#E5E7EB]">
            {reminders.map(r => (
              <div key={r.id} className="flex items-center justify-between py-4 group">
                <div>
                  <p className="text-[15px] font-bold text-[#111827] mb-1">{r.drugName}</p>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
                    <span className="text-[12px] text-[#6B7280] flex items-center gap-1">
                      <Clock className="w-3 h-3" />{r.time}
                    </span>
                    <span className="text-[12px] text-[#D1D5DB]">·</span>
                    <span className="text-[12px] text-[#6B7280]">{r.frequency}</span>
                    <span className="text-[12px] text-[#D1D5DB]">·</span>
                    <span className="text-[12px] text-[#6B7280] flex items-center gap-1">
                      <Phone className="w-3 h-3" />{r.phone}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => removeReminder(r.id)}
                  className="opacity-0 group-hover:opacity-100 w-7 h-7 flex items-center justify-center rounded-lg text-[#D1D5DB] hover:text-red-400 transition-all"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── History ──────────────────────────────────────────────────────────────────

function HistoryPage({ history, navigate }: { history: HistoryEntry[]; navigate: (p: Page) => void }) {
  return (
    <div className="max-w-[660px] mx-auto px-6 py-12">
      <div className="flex items-end justify-between mb-10">
        <div>
          <h1 className="text-[2rem] font-extrabold text-[#111827] mb-1.5">Saved Reports</h1>
          <p className="text-[15px] text-[#6B7280]">All your simplified prescriptions and lab reports.</p>
        </div>
        <button
          onClick={() => navigate("upload")}
          className="flex items-center gap-1.5 text-[13px] font-bold text-[#0D9488] hover:text-[#0b8277] transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          New Upload
        </button>
      </div>

      <div className="divide-y divide-[#E5E7EB]">
        {history.map(entry => (
          <div key={entry.id} className="flex items-center justify-between py-5 group">
            <div className="flex items-start gap-5">
              {/* Type indicator */}
              <div className="pt-0.5">
                <span className={`text-[10px] font-extrabold tracking-[0.1em] uppercase ${
                  entry.type === "Lab Report" ? "text-[#D97706]" : "text-[#0D9488]"
                }`}>
                  {entry.type === "Lab Report" ? "Lab" : "Rx"}
                </span>
              </div>
              <div>
                <p className="text-[15.5px] font-bold text-[#111827] mb-1">{entry.patient}</p>
                <div className="flex items-center gap-3">
                  <span className="text-[12px] text-[#9CA3AF] flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    {entry.date}
                  </span>
                  <span className="text-[12px] text-[#D1D5DB]">·</span>
                  <span className="text-[12px] text-[#9CA3AF]">
                    {entry.medicines} medicine{entry.medicines !== 1 ? "s" : ""}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-1">
              <button
                onClick={() => navigate("results")}
                className="flex items-center gap-0.5 text-[13px] text-[#9CA3AF] font-semibold px-2.5 py-1.5 rounded-lg hover:text-[#0D9488] hover:bg-[#E6F4F1] transition-all opacity-0 group-hover:opacity-100"
              >
                View
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
              <button className="flex items-center gap-1 text-[13px] text-[#9CA3AF] font-semibold px-2.5 py-1.5 rounded-lg hover:text-[#111827] transition-all border border-transparent hover:border-[#E5E7EB]">
                <Download className="w-3.5 h-3.5" />
                PDF
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
