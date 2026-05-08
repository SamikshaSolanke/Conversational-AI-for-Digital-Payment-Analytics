import React, { useState, useEffect, useRef } from 'react';
import WhatIfLab from './WhatIfLab';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid,
  LineChart, Line, PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis
} from 'recharts';
import {
  BarChart3,
  MessageSquare,
  Settings,
  Database,
  Search,
  ChevronRight,
  Send,
  User,
  Bot,
  FlaskConical
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000/api';

const Sidebar = ({ activeView, setActiveView }: { activeView: string; setActiveView: (v: string) => void }) => (
  <aside className="w-16 lg:w-64 bg-slate-900 border-r border-slate-800 text-slate-300 flex flex-col h-screen fixed left-0 top-0">
    <div className="h-16 flex items-center justify-center lg:justify-start lg:px-6 border-b border-slate-800">
      <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center flex-shrink-0">
        <Database className="w-5 h-5 text-white" />
      </div>
      <span className="ml-3 font-semibold text-white text-lg hidden lg:block tracking-tight">InsightX</span>
    </div>

    <nav className="flex-1 py-6 flex flex-col gap-2 px-3">
      <button onClick={() => setActiveView('chat')} className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors w-full text-left ${
        activeView === 'chat' ? 'bg-blue-500/10 text-blue-400' : 'hover:bg-slate-800'
      }`}>
        <MessageSquare className="w-5 h-5" />
        <span className="hidden lg:block font-medium">Conversational Analytics</span>
      </button>
      <button onClick={() => setActiveView('whatif')} className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors w-full text-left ${
        activeView === 'whatif' ? 'bg-violet-500/10 text-violet-400' : 'hover:bg-slate-800'
      }`}>
        <FlaskConical className="w-5 h-5" />
        <span className="hidden lg:block font-medium">What-If Lab</span>
      </button>
      <button onClick={() => setActiveView('chat')} className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-800 transition-colors w-full text-left">
        <BarChart3 className="w-5 h-5" />
        <span className="hidden lg:block font-medium">Dashboards</span>
      </button>
    </nav>

    <div className="p-4 border-t border-slate-800">
      <a href="#" className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-slate-800 transition-colors">
        <Settings className="w-5 h-5" />
        <span className="hidden lg:block font-medium">Settings</span>
      </a>
    </div>
  </aside>
);

const TopNav = () => (
  <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 pl-[4.5rem] lg:pl-72 fixed top-0 right-0 left-0 z-10">
    <div className="flex items-center gap-2">
      <h1 className="text-xl font-semibold text-slate-800">Decision Intelligence</h1>
      <ChevronRight className="w-4 h-4 text-slate-400" />
      <span className="text-slate-500 font-medium">UPI Analysis 2024</span>
    </div>
    <div className="flex items-center gap-4">
      <div className="relative hidden md:block">
        <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
        <input
          type="text"
          placeholder="Search..."
          className="pl-9 pr-4 py-2 bg-slate-100 border-transparent focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-200 rounded-lg text-sm w-64 transition-all outline-none"
        />
      </div>
      <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 font-bold border border-indigo-200">
        U
      </div>
    </div>
  </header>
);

const SparklesIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500">
    <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" />
  </svg>
)

type Message = {
  id: string;
  sender: 'user' | 'ai';
  text?: string;
  visualization?: {
    type: string;
    chart_type: string;
    data: any[];
    title: string;
  };
}

const MAGMA_COLORS = ['#1d1137', '#341559', '#531b71', '#761e7e', '#9b227c', '#c02d71', '#e0425e', '#f56346', '#fd8d34', '#fbc043'];
const VIRIDIS_COLORS = ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21908c', '#27ad81', '#5dc863', '#aadc32', '#fde725'];

const CustomHeatmap = ({ data }: { data: any[] }) => {
  // Sort numerically if possible, otherwise alphabetically
  const sortAxis = (arr: any[]) => arr.sort((a, b) => (Number(a) - Number(b)) || String(a).localeCompare(String(b)));

  const xValues = sortAxis(Array.from(new Set(data.map(d => d.x))));
  const yValues = sortAxis(Array.from(new Set(data.map(d => d.y)))).reverse(); // Reverse Y so high numbers/letters are top

  const values = data.map(d => d.value);
  const min = Math.min(...values);
  const max = Math.max(...values);

  return (
    <div className="w-full h-full flex flex-col pt-2 pr-4 pb-8">
      <div className="flex-1 flex flex-col min-h-0">
        {yValues.map(y => (
          <div key={y} className="flex flex-1 min-h-0 group/row">
            <div className="w-24 md:w-32 shrink-0 text-right pr-4 font-medium text-slate-500 text-[10px] md:text-sm self-center truncate group-hover/row:text-slate-800 transition-colors" title={y}>{y}</div>
            {xValues.map(x => {
              const cell = data.find(d => d.x === x && d.y === y);
              const val = cell ? cell.value : 0;
              const ratio = max > min ? (val - min) / (max - min) : 0;
              // Clean seaborn-style teal opacity gradient
              const bg = cell ? `rgba(21, 94, 117, ${0.1 + ratio * 0.9})` : '#f8fafc';

              return (
                <div
                  key={x}
                  className="flex-1 m-[1px] rounded flex items-center justify-center relative group transition-all hover:scale-[1.05] hover:shadow-md hover:z-10 border border-slate-900/5 min-w-0"
                  style={{ backgroundColor: bg }}
                >
                  {xValues.length <= 25 && (
                    <span className={`text-[10px] sm:text-xs font-semibold opacity-0 sm:opacity-100 ${ratio > 0.4 ? 'text-white' : 'text-slate-700'}`}>
                      {Number.isInteger(val) ? val : Number(val).toFixed(2)}
                    </span>
                  )}
                  <div className="opacity-0 group-hover:opacity-100 absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-slate-900 text-white z-20 px-3 py-1.5 rounded-lg text-xs whitespace-nowrap shadow-xl pointer-events-none font-medium">
                    {y} × {x}: <span className="text-blue-300 ml-1">{Number.isInteger(val) ? val : Number(val).toFixed(2)}</span>
                    <svg className="absolute text-slate-900 h-2 w-full left-0 top-full" x="0px" y="0px" viewBox="0 0 255 255"><polygon className="fill-current" points="0,0 127.5,127.5 255,0" /></svg>
                  </div>
                </div>
              )
            })}
          </div>
        ))}
      </div>
      <div className="flex mt-3 h-20 shrink-0">
        <div className="w-24 md:w-32 shrink-0"></div>
        {xValues.map(x => (
          <div key={x} className="flex-1 relative min-w-0">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 -rotate-45 whitespace-nowrap text-[10px] sm:text-xs font-medium text-slate-500 hover:text-slate-800 origin-top-left transition-colors" title={x}>{x}</div>
          </div>
        ))}
      </div>
    </div>
  );
};


const ChatPanel = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      sender: 'ai',
      text: "Hello! I'm your AI Data Analyst. I can help answer questions about your 250k UPI transactions dataset. Try asking:\n\n- \"What is the overall success rate?\"\n- \"Which age group has the most failed transactions?\""
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { id: Date.now().toString(), sender: 'user', text: input.trim() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await axios.post(`${API_BASE_URL}/chat`, { query: userMsg.text });

      // The backend returns a 200 OK with `{"error": ...}` instead of throwing an HTTP Exception sometimes
      if (res.data.error) {
        throw new Error(res.data.error);
      }

      let aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
      };

      if (typeof res.data.response === 'string') {
        aiMsg.text = res.data.response;
      } else if (res.data.response?.type === 'visualization') {
        aiMsg.visualization = res.data.response;
        aiMsg.text = res.data.response.text || `Here is the visualization for: **${res.data.response.title}**`;
      } else {
        aiMsg.text = "Received an unknown response format.";
      }

      setMessages(prev => [...prev, aiMsg]);
    } catch (error: any) {
      console.error("Chat Error:", error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        text: `Error connecting to AI: ${error.message}. (Did you set your GROQ_API_KEY in the .env file?)`
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-50 border-r border-slate-200 overflow-hidden">
      {/* Chat Header */}
      <div className="px-6 py-4 border-b border-slate-200 bg-white/50 backdrop-blur-sm sticky top-0 z-10 flex justify-between items-center">
        <div>
          <h2 className="font-semibold text-slate-800 flex items-center gap-2">
            <SparklesIcon /> Data Assistant
          </h2>
          <p className="text-sm text-slate-500">Ask questions about your UPI dataset</p>
        </div>
        {isLoading && (
          <div className="flex items-center gap-2 text-blue-500 text-sm font-medium">
            <div className="w-4 h-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></div>
            Analyzing...
          </div>
        )}
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex items-start gap-4 ${msg.sender === 'user' ? 'flex-row-reverse' : ''}`}>
            {msg.sender === 'ai' ? (
              <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
                <Bot className="w-5 h-5 text-blue-600" />
              </div>
            ) : (
              <div className="w-8 h-8 rounded-lg bg-indigo-100 flex items-center justify-center flex-shrink-0 mt-1 text-indigo-700 font-bold text-sm">
                U
              </div>
            )}
            <div className={`${msg.sender === 'user'
              ? 'bg-blue-600 text-white rounded-2xl rounded-tr-sm'
              : 'bg-white border border-slate-200 text-slate-700 rounded-2xl rounded-tl-sm'
              } p-4 shadow-sm max-w-[85%] whitespace-pre-wrap`}
            >
              {msg.text && <div>{msg.text}</div>}
              {msg.visualization && (
                <div className="mt-4 h-96 w-full min-w-[300px] bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
                  {msg.visualization.chart_type === 'heatmap' ? (
                    <CustomHeatmap data={msg.visualization.data} />
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      {msg.visualization.chart_type === 'bar' ? (
                        msg.visualization.data.length > 5 || msg.visualization.data.some((d: any) => String(d.name).length > 8) ? (
                          <BarChart data={msg.visualization.data} layout="vertical" margin={{ top: 10, right: 30, left: 40, bottom: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                            <XAxis type="number" tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                            <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#475569' }} width={80} axisLine={false} tickLine={false} />
                            <Tooltip cursor={{ fill: '#F1F5F9' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                            <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={40}>
                              {msg.visualization.data.map((entry: any, index: number) => (
                                <Cell key={`cell-${index}`} fill={MAGMA_COLORS[index % MAGMA_COLORS.length]} />
                              ))}
                            </Bar>
                          </BarChart>
                        ) : (
                          <BarChart data={msg.visualization.data} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                            <XAxis dataKey="name" angle={0} tick={{ fontSize: 12, fill: '#475569' }} axisLine={false} tickLine={false} />
                            <YAxis tick={{ fontSize: 12, fill: '#475569' }} axisLine={false} tickLine={false} />
                            <Tooltip cursor={{ fill: '#F1F5F9' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={60}>
                              {msg.visualization.data.map((entry: any, index: number) => (
                                <Cell key={`cell-${index}`} fill={VIRIDIS_COLORS[index % VIRIDIS_COLORS.length]} />
                              ))}
                            </Bar>
                          </BarChart>
                        )
                      ) : msg.visualization.chart_type === 'line' ? (
                        <LineChart data={msg.visualization.data} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                          <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                          <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                          <Line type="monotone" dataKey="value" stroke="#3b528b" strokeWidth={3} dot={{ fill: '#3b528b', strokeWidth: 2, r: 4 }} activeDot={{ r: 6 }} />
                        </LineChart>
                      ) : msg.visualization.chart_type === 'pie' ? (
                        <PieChart margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                          <Pie data={msg.visualization.data} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={2} label={({ name, percent }) => `${name} ${((percent || 0) * 100).toFixed(0)}%`}>
                            {msg.visualization.data.map((entry: any, index: number) => (
                              <Cell key={`cell-${index}`} fill={MAGMA_COLORS[index % MAGMA_COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                        </PieChart>
                      ) : msg.visualization.chart_type === 'scatter' ? (
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                          <XAxis type="number" dataKey="x" name="X" tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                          <YAxis type="number" dataKey="y" name="Y" tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                          <Scatter name="Data" data={msg.visualization.data} fill="#2c728e" />
                        </ScatterChart>
                      ) : (
                        <div className="flex items-center justify-center h-full text-slate-400">Unsupported chart type</div>
                      )}
                    </ResponsiveContainer>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <div className="p-4 bg-white border-t border-slate-200">
        <form
          className="relative flex items-center"
          onSubmit={(e) => { e.preventDefault(); handleSend(); }}
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
            placeholder="Ask a question about your data..."
            className="w-full pl-4 pr-12 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-100 outline-none transition-all disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="absolute right-2 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
        <p className="text-center text-xs text-slate-400 mt-2">AI can make mistakes. Verify important data.</p>
      </div>
    </div>
  );
};

const VisPanel = () => {
  const [whatIfData, setWhatIfData] = useState<any[]>([]);
  const [clusterData, setClusterData] = useState<any[]>([]);
  const [summaryData, setSummaryData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Fetch existing summary (from step 1) if desired. We use dummy values otherwise.
        const summaryRes = await axios.get(`${API_BASE_URL}/summary`);
        setSummaryData(summaryRes.data);

        // Fetch What-If Simulation for 5G
        const whatIfRes = await axios.post(`${API_BASE_URL}/what-if`, { target_network: '5G' });

        if (!whatIfRes.data.error) {
          setWhatIfData([
            { name: "Before (3G/WiFi)", rate: whatIfRes.data.baseline_success_rate_percent },
            { name: "After (simulated 5G)", rate: whatIfRes.data.simulated_success_rate_percent }
          ]);
        }

        // Fetch Clusters
        const clusterRes = await axios.post(`${API_BASE_URL}/clusters`);
        if (!clusterRes.data.error) {
          // Mapping cluster data into Scatter format
          const mappedClusters = clusterRes.data.clusters.map((c: any) => ({
            cluster: `Cluster ${c.cluster_id}`,
            amount: c.avg_amount_inr,
            hour: c.avg_hour_of_day,
            size: c.transaction_count
          }));
          setClusterData(mappedClusters);
        }

      } catch (error) {
        console.error("Error fetching visualization data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="w-full lg:w-[450px] xl:w-[550px] bg-slate-50 p-6 overflow-y-auto hidden md:flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-slate-800 text-lg">Statistical Insights</h2>
        <button
          onClick={() => window.location.reload()}
          className="text-sm font-medium text-blue-600 hover:text-blue-700 bg-blue-50 px-3 py-1.5 rounded-md"
        >
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="w-8 h-8 rounded-full border-4 border-slate-200 border-t-blue-500 animate-spin"></div>
        </div>
      ) : (
        <>
          {/* Stats Cards */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
              <div className="text-slate-500 text-sm font-medium mb-1 flex items-center gap-2">
                Total Transactions
              </div>
              <div className="text-3xl font-bold text-slate-800">
                {summaryData?.total_transactions ? (summaryData.total_transactions / 1000).toFixed(1) + 'k' : '---'}
              </div>
              <div className="text-xs text-green-600 font-medium mt-2 flex items-center bg-green-50 w-fit px-2 py-0.5 rounded-full">
                All data parsed
              </div>
            </div>
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
              <div className="text-slate-500 text-sm font-medium mb-1">Overall Success Rate</div>
              <div className="text-3xl font-bold text-slate-800">
                {summaryData?.overall_success_rate ? summaryData.overall_success_rate + '%' : '---'}
              </div>
              <div className="text-xs text-emerald-600 font-medium mt-2 flex items-center bg-emerald-50 w-fit px-2 py-0.5 rounded-full">
                Healthy Status
              </div>
            </div>
          </div>

          {/* What-if BarChart */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col h-72">
            <h3 className="font-semibold text-slate-800 flex items-center gap-2 mb-4">
              What-If Simulation: Network Upgrade
            </h3>
            <div className="flex-1">
              {whatIfData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={whatIfData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#64748B', fontSize: 12 }} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fill: '#64748B', fontSize: 12 }} domain={['auto', 'auto']} />
                    <Tooltip cursor={{ fill: '#F1F5F9' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                    <Bar dataKey="rate" fill="#3B82F6" radius={[4, 4, 0, 0]} maxBarSize={60} name="Success Rate (%)" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-sm text-slate-400 border-2 border-dashed border-slate-100 rounded-lg">No prediction data available.</div>
              )}
            </div>
          </div>

          {/* Clusters ScatterPlot */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col h-72">
            <h3 className="font-semibold text-slate-800 flex items-center gap-2 mb-4">
              K-Means Segments (Avg Amount vs Hour)
            </h3>
            <div className="flex-1">
              {clusterData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                    <XAxis type="number" dataKey="hour" name="Hour of Day" axisLine={false} tickLine={false} tick={{ fill: '#64748B', fontSize: 12 }} domain={[0, 24]} />
                    <YAxis type="number" dataKey="amount" name="Avg Amount (INR)" axisLine={false} tickLine={false} tick={{ fill: '#64748B', fontSize: 12 }} />
                    <ZAxis type="number" dataKey="size" range={[100, 1000]} name="Transaction Count" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                    <Scatter name="Clusters" data={clusterData} fill="#8B5CF6" fillOpacity={0.7} />
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-sm text-slate-400 border-2 border-dashed border-slate-100 rounded-lg">No cluster data available.</div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

function App() {
  const [activeView, setActiveView] = useState('chat');

  return (
    <div className="h-screen w-full flex bg-slate-50 font-sans text-slate-900 overflow-hidden">
      <Sidebar activeView={activeView} setActiveView={setActiveView} />
      <TopNav />
      {/* Main Content Area */}
      <main className="flex-1 flex ml-16 lg:ml-64 mt-16 h-[calc(100vh-4rem)]">
        {activeView === 'chat' ? (
          <>
            <ChatPanel />
            <VisPanel />
          </>
        ) : (
          <WhatIfLab />
        )}
      </main>
    </div>
  );
}

export default App;
