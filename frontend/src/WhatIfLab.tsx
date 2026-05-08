import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  Cell, PieChart, Pie, Legend
} from 'recharts';
import { FlaskConical, RotateCcw, Play, TrendingUp, TrendingDown, Minus, ShieldAlert, DollarSign, CheckCircle2, Sparkles, Save, Layers, Lightbulb, Wand2, X, ArrowRight } from 'lucide-react';

const API = 'http://localhost:8000/api';

type ParamOption = {
  name: string;
  type: 'categorical' | 'numerical';
  values?: any[];
  range?: { min: number; max: number; step: number };
  baseline: any;
};

type SimResult = {
  success_rate: { baseline: number; simulated: number; delta: number; unit: string; model_metrics?: { accuracy: number; cv_accuracy: number; precision: number; recall: number } };
  fraud_risk: { baseline: number; simulated: number; delta: number; unit: string; model_metrics?: { accuracy: number; cv_accuracy: number; precision: number; recall: number } };
  expected_amount: { baseline: number; simulated: number; delta: number; unit: string; model_r2?: number };
  feature_importances: Record<string, { feature: string; importance: number }[]>;
  overrides_applied: Record<string, any>;
};

type SensitivityItem = { parameter: string; min_value: number; max_value: number; range: number; baseline: number };

type SavedScenario = { name: string; overrides: Record<string, any>; result: SimResult };

type RecommendResult = {
  target: string; goal: string; baseline: number;
  best: { overrides: Record<string, any>; predicted: number; delta: number; summary: string }[];
  worst: { overrides: Record<string, any>; predicted: number; delta: number; summary: string };
  explanation: string;
};

const GRADIENT_COLORS = ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#818cf8', '#6d28d9', '#7c3aed', '#4f46e5', '#4338ca', '#3730a3'];

const LABEL_MAP: Record<string, string> = {
  network_type: 'Network Type',
  device_type: 'Device Type',
  merchant_category: 'Merchant Category',
  sender_age_group: 'Age Group',
  sender_state: 'State',
  sender_bank: 'Sender Bank',
  'transaction type': 'Transaction Type',
  hour_of_day: 'Hour of Day',
  'amount (INR)': 'Amount (INR)',
  is_weekend: 'Weekend',
};

const prettyLabel = (s: string) => LABEL_MAP[s] || s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

// Delta indicator component
const DeltaChip = ({ value, invert = false }: { value: number; invert?: boolean }) => {
  const isPositive = invert ? value < 0 : value > 0;
  const isNeutral = Math.abs(value) < 0.01;
  const Icon = isNeutral ? Minus : isPositive ? TrendingUp : TrendingDown;
  const color = isNeutral ? 'text-slate-400 bg-slate-50' : isPositive ? 'text-emerald-600 bg-emerald-50' : 'text-red-500 bg-red-50';
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold ${color}`}>
      <Icon className="w-3 h-3" />
      {value > 0 ? '+' : ''}{value.toFixed(2)}
    </span>
  );
};

export default function WhatIfLab() {
  const [options, setOptions] = useState<ParamOption[]>([]);
  const [overrides, setOverrides] = useState<Record<string, any>>({});
  const [result, setResult] = useState<SimResult | null>(null);
  const [sensitivity, setSensitivity] = useState<SensitivityItem[]>([]);
  const [sensTarget, setSensTarget] = useState('success_rate');
  const [loading, setLoading] = useState(false);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [hasSimulated, setHasSimulated] = useState(false);

  // NL query state
  const [nlQuery, setNlQuery] = useState('');
  const [nlLoading, setNlLoading] = useState(false);
  const [nlInterpretation, setNlInterpretation] = useState('');

  // Scenario comparison state
  const [savedScenarios, setSavedScenarios] = useState<SavedScenario[]>([]);

  // AI Recommendation state
  const [recResult, setRecResult] = useState<RecommendResult | null>(null);
  const [recLoading, setRecLoading] = useState(false);
  const [recTarget, setRecTarget] = useState('success_rate');
  const [recGoal, setRecGoal] = useState('maximize');

  // Fetch available options on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await axios.get(`${API}/what-if/options`);
        if (!res.data.error) setOptions(res.data.parameters || []);
      } catch (e) { console.error(e); }
      finally { setOptionsLoading(false); }
    })();
  }, []);

  const handleOverride = (name: string, value: any) => {
    setOverrides(prev => {
      const baseline = options.find(o => o.name === name)?.baseline;
      if (value === '' || value === baseline || value === String(baseline)) {
        const next = { ...prev };
        delete next[name];
        return next;
      }
      return { ...prev, [name]: value };
    });
  };

  const runSimulation = useCallback(async () => {
    setLoading(true);
    try {
      const [simRes, sensRes] = await Promise.all([
        axios.post(`${API}/what-if/simulate`, { overrides }),
        axios.post(`${API}/what-if/sensitivity`, { target: sensTarget }),
      ]);
      if (!simRes.data.error) setResult(simRes.data);
      if (!sensRes.data.error) setSensitivity(sensRes.data.sensitivity || []);
      setHasSimulated(true);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, [overrides, sensTarget]);

  const resetAll = () => {
    setOverrides({});
    setResult(null);
    setSensitivity([]);
    setHasSimulated(false);
    setNlInterpretation('');
  };

  // NL query handler
  const handleNlQuery = async () => {
    if (!nlQuery.trim() || nlLoading) return;
    setNlLoading(true);
    setNlInterpretation('');
    try {
      const res = await axios.post(`${API}/what-if/natural`, { query: nlQuery });
      if (res.data.error) { setNlInterpretation(`Error: ${res.data.error}`); return; }
      setOverrides(res.data.parsed_overrides || {});
      setResult(res.data.simulation);
      setNlInterpretation(res.data.interpretation);
      setHasSimulated(true);
      // Also fetch sensitivity
      const sensRes = await axios.post(`${API}/what-if/sensitivity`, { target: sensTarget });
      if (!sensRes.data.error) setSensitivity(sensRes.data.sensitivity || []);
    } catch (e) { console.error(e); setNlInterpretation('Failed to process query.'); }
    finally { setNlLoading(false); }
  };

  // Scenario save/delete
  const saveScenario = () => {
    if (!result || savedScenarios.length >= 5) return;
    const name = `Scenario ${String.fromCharCode(65 + savedScenarios.length)}`;
    setSavedScenarios(prev => [...prev, { name, overrides: { ...overrides }, result }]);
  };
  const deleteScenario = (idx: number) => {
    setSavedScenarios(prev => prev.filter((_, i) => i !== idx));
  };

  // AI Recommendation handler
  const fetchRecommendations = async () => {
    setRecLoading(true);
    try {
      const res = await axios.post(`${API}/what-if/recommend`, { target: recTarget, goal: recGoal });
      if (!res.data.error) setRecResult(res.data);
    } catch (e) { console.error(e); }
    finally { setRecLoading(false); }
  };

  // Apply a recommendation's overrides
  const applyRecommendation = async (rec: Record<string, any>) => {
    setOverrides(rec);
    setLoading(true);
    try {
      const [simRes, sensRes] = await Promise.all([
        axios.post(`${API}/what-if/simulate`, { overrides: rec }),
        axios.post(`${API}/what-if/sensitivity`, { target: sensTarget }),
      ]);
      if (!simRes.data.error) setResult(simRes.data);
      if (!sensRes.data.error) setSensitivity(sensRes.data.sensitivity || []);
      setHasSimulated(true);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  // Build comparison chart data
  const comparisonData = result ? [
    { name: 'Success Rate (%)', Baseline: result.success_rate.baseline, Simulated: result.success_rate.simulated },
    { name: 'Fraud Risk (%)', Baseline: result.fraud_risk.baseline, Simulated: result.fraud_risk.simulated },
  ] : [];

  // Build feature importance data for donut
  const importanceData = result?.feature_importances?.success_rate?.slice(0, 8).map((f, i) => ({
    name: prettyLabel(f.feature),
    value: +(f.importance * 100).toFixed(1),
  })) || [];

  // Tornado chart data
  const tornadoData = sensitivity.slice(0, 8).map(s => ({
    name: prettyLabel(s.parameter),
    low: +(s.min_value - s.baseline).toFixed(2),
    high: +(s.max_value - s.baseline).toFixed(2),
    range: s.range,
  }));

  if (optionsLoading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-slate-50">
        <div className="text-center">
          <div className="w-10 h-10 rounded-full border-4 border-slate-200 border-t-indigo-500 animate-spin mx-auto mb-4"></div>
          <p className="text-slate-500 font-medium">Loading What-If Engine...</p>
          <p className="text-slate-400 text-sm mt-1">Training prediction models on your dataset</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col lg:flex-row bg-slate-50 overflow-hidden">
      {/* Left: Parameter Controls */}
      <div className="w-full lg:w-[340px] xl:w-[380px] bg-white border-r border-slate-200 flex flex-col overflow-y-auto">
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-200 bg-gradient-to-r from-indigo-50 to-violet-50">
          <h2 className="font-bold text-slate-800 flex items-center gap-2 text-lg">
            <FlaskConical className="w-5 h-5 text-indigo-600" /> What-If Lab
          </h2>
          <p className="text-sm text-slate-500 mt-1">Adjust parameters to predict outcomes</p>
        </div>

        {/* NL Query Input */}
        <div className="px-5 py-3 border-b border-slate-200 bg-gradient-to-r from-violet-50/50 to-indigo-50/50">
          <form onSubmit={(e) => { e.preventDefault(); handleNlQuery(); }} className="relative">
            <Wand2 className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-violet-400" />
            <input
              type="text"
              value={nlQuery}
              onChange={e => setNlQuery(e.target.value)}
              disabled={nlLoading}
              placeholder='e.g. "What if network is 5G and device is iOS?"'
              className="w-full pl-9 pr-10 py-2 bg-white border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-violet-200 focus:border-violet-400 outline-none transition-all disabled:opacity-50"
            />
            <button type="submit" disabled={!nlQuery.trim() || nlLoading}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1.5 bg-violet-600 text-white rounded-md hover:bg-violet-700 disabled:opacity-50 transition-all">
              {nlLoading ? <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" /> : <Sparkles className="w-3 h-3" />}
            </button>
          </form>
          {nlInterpretation && (
            <p className="text-xs text-violet-600 mt-1.5 font-medium">{nlInterpretation}</p>
          )}
        </div>

        {/* Controls */}
        <div className="flex-1 px-5 py-4 space-y-4 overflow-y-auto">
          {options.map(opt => (
            <div key={opt.name} className="group">
              <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
                {prettyLabel(opt.name)}
                {overrides[opt.name] !== undefined && (
                  <span className="ml-2 text-indigo-500 normal-case tracking-normal">● modified</span>
                )}
              </label>

              {opt.type === 'categorical' && opt.values && (
                <select
                  value={overrides[opt.name] ?? opt.baseline}
                  onChange={e => handleOverride(opt.name, e.target.value)}
                  className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-200 focus:border-indigo-400 outline-none transition-all hover:border-slate-300"
                >
                  {opt.values.map((v: any) => (
                    <option key={v} value={v}>
                      {v} {v === opt.baseline ? '(baseline)' : ''}
                    </option>
                  ))}
                </select>
              )}

              {opt.type === 'numerical' && opt.name === 'is_weekend' && (
                <div className="flex gap-2">
                  {[{ label: 'Weekday', val: 0 }, { label: 'Weekend', val: 1 }].map(btn => (
                    <button
                      key={btn.val}
                      onClick={() => handleOverride(opt.name, btn.val)}
                      className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition-all ${
                        (overrides[opt.name] ?? opt.baseline) === btn.val
                          ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                          : 'bg-slate-50 text-slate-600 border-slate-200 hover:border-slate-300'
                      }`}
                    >
                      {btn.label}
                    </button>
                  ))}
                </div>
              )}

              {opt.type === 'numerical' && opt.range && opt.name !== 'is_weekend' && (
                <div>
                  <input
                    type="range"
                    min={opt.range.min}
                    max={opt.range.max}
                    step={opt.range.step}
                    value={overrides[opt.name] ?? opt.baseline}
                    onChange={e => handleOverride(opt.name, Number(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                  />
                  <div className="flex justify-between text-xs text-slate-400 mt-1">
                    <span>{opt.range.min}</span>
                    <span className="font-semibold text-indigo-600">{overrides[opt.name] ?? opt.baseline}</span>
                    <span>{opt.range.max}</span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="p-4 border-t border-slate-200 bg-slate-50 space-y-2">
          <button
            onClick={runSimulation}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 text-white rounded-xl font-semibold hover:bg-indigo-700 transition-all shadow-sm disabled:opacity-50 disabled:cursor-wait"
          >
            {loading ? (
              <><div className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin"></div> Simulating...</>
            ) : (
              <><Play className="w-4 h-4" /> Run Simulation</>
            )}
          </button>
          
          {hasSimulated && (
            <button
              onClick={saveScenario}
              disabled={savedScenarios.length >= 5}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-white text-indigo-600 border border-indigo-200 rounded-xl font-medium hover:bg-indigo-50 transition-all text-sm disabled:opacity-50"
            >
              <Save className="w-4 h-4" /> Save Scenario {savedScenarios.length >= 5 ? '(Max 5)' : ''}
            </button>
          )}

          <button
            onClick={resetAll}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 text-slate-500 rounded-xl font-medium hover:bg-slate-100 transition-all text-sm"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Reset to Baseline
          </button>
        </div>
      </div>

      {/* Right: Results Dashboard */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {!hasSimulated ? (
          <div className="flex-1 flex items-center justify-center h-full min-h-[400px]">
            <div className="text-center max-w-md">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-100 to-violet-100 flex items-center justify-center mx-auto mb-6">
                <FlaskConical className="w-10 h-10 text-indigo-500" />
              </div>
              <h3 className="text-xl font-bold text-slate-800 mb-2">Ready to Simulate</h3>
              <p className="text-slate-500 leading-relaxed">
                Adjust the parameters on the left, then click <strong>"Run Simulation"</strong> to see
                how changes affect success rate, fraud risk, and expected transaction amount.
              </p>
              <div className="flex items-center justify-center gap-4 mt-6">
                {[
                  { icon: CheckCircle2, label: 'Success Rate', color: 'text-emerald-500' },
                  { icon: ShieldAlert, label: 'Fraud Risk', color: 'text-amber-500' },
                  { icon: DollarSign, label: 'Amount', color: 'text-blue-500' },
                ].map(m => (
                  <div key={m.label} className="flex items-center gap-1.5 text-sm text-slate-500">
                    <m.icon className={`w-4 h-4 ${m.color}`} />{m.label}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : result ? (
          <>
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Success Rate */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-lg bg-emerald-100 flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  </div>
                  <span className="text-sm font-semibold text-slate-600">Success Rate</span>
                </div>
                <div className="flex items-end gap-3">
                  <div>
                    <div className="text-3xl font-bold text-slate-800">{result.success_rate.simulated}%</div>
                    <div className="text-xs text-slate-400 mt-1">Baseline: {result.success_rate.baseline}%</div>
                  </div>
                  <DeltaChip value={result.success_rate.delta} />
                </div>
                {result.success_rate.model_metrics && (
                  <div className="mt-4 pt-3 border-t border-slate-100 grid grid-cols-2 gap-y-1 gap-x-2 text-[10px] text-slate-500">
                    <div className="flex justify-between"><span>CV Acc:</span> <span className="font-medium text-slate-700">{result.success_rate.model_metrics.cv_accuracy}%</span></div>
                    <div className="flex justify-between"><span>Prec:</span> <span className="font-medium text-slate-700">{result.success_rate.model_metrics.precision}%</span></div>
                    <div className="flex justify-between"><span>Base Acc:</span> <span className="font-medium text-slate-700">{result.success_rate.model_metrics.accuracy}%</span></div>
                    <div className="flex justify-between"><span>Recall:</span> <span className="font-medium text-slate-700">{result.success_rate.model_metrics.recall}%</span></div>
                  </div>
                )}
              </div>

              {/* Fraud Risk */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-lg bg-amber-100 flex items-center justify-center">
                    <ShieldAlert className="w-4 h-4 text-amber-600" />
                  </div>
                  <span className="text-sm font-semibold text-slate-600">Fraud Risk</span>
                </div>
                <div className="flex items-end gap-3">
                  <div>
                    <div className="text-3xl font-bold text-slate-800">{result.fraud_risk.simulated}%</div>
                    <div className="text-xs text-slate-400 mt-1">Baseline: {result.fraud_risk.baseline}%</div>
                  </div>
                  <DeltaChip value={result.fraud_risk.delta} invert />
                </div>
                {result.fraud_risk.model_metrics && (
                  <div className="mt-4 pt-3 border-t border-slate-100 grid grid-cols-2 gap-y-1 gap-x-2 text-[10px] text-slate-500">
                    <div className="flex justify-between"><span>CV Acc:</span> <span className="font-medium text-slate-700">{result.fraud_risk.model_metrics.cv_accuracy}%</span></div>
                    <div className="flex justify-between"><span>Prec:</span> <span className="font-medium text-slate-700">{result.fraud_risk.model_metrics.precision}%</span></div>
                    <div className="flex justify-between"><span>Base Acc:</span> <span className="font-medium text-slate-700">{result.fraud_risk.model_metrics.accuracy}%</span></div>
                    <div className="flex justify-between"><span>Recall:</span> <span className="font-medium text-slate-700">{result.fraud_risk.model_metrics.recall}%</span></div>
                  </div>
                )}
              </div>

              {/* Expected Amount */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center">
                    <DollarSign className="w-4 h-4 text-blue-600" />
                  </div>
                  <span className="text-sm font-semibold text-slate-600">Expected Amount</span>
                </div>
                <div className="flex items-end gap-3">
                  <div>
                    <div className="text-3xl font-bold text-slate-800">₹{result.expected_amount.simulated.toLocaleString()}</div>
                    <div className="text-xs text-slate-400 mt-1">Baseline: ₹{result.expected_amount.baseline.toLocaleString()}</div>
                  </div>
                  <DeltaChip value={result.expected_amount.delta} />
                </div>
                {result.expected_amount.model_r2 !== undefined && (
                  <div className="text-[10px] text-slate-400 mt-2">Model R²: {result.expected_amount.model_r2}</div>
                )}
              </div>
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Comparison Bar Chart */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <h3 className="font-semibold text-slate-800 mb-4">Baseline vs Simulated</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                      <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                      <Legend />
                      <Bar dataKey="Baseline" fill="#94a3b8" radius={[4, 4, 0, 0]} maxBarSize={50} />
                      <Bar dataKey="Simulated" fill="#6366f1" radius={[4, 4, 0, 0]} maxBarSize={50} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Feature Importance Donut */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <h3 className="font-semibold text-slate-800 mb-4">Feature Importance (Success Rate)</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={importanceData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={55} outerRadius={90} paddingAngle={2}
                        label={({ name, value }) => `${name}: ${value}%`}
                      >
                        {importanceData.map((_, i) => (
                          <Cell key={i} fill={GRADIENT_COLORS[i % GRADIENT_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Sensitivity / Tornado Chart */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-slate-800">Parameter Sensitivity Analysis</h3>
                <select
                  value={sensTarget}
                  onChange={e => setSensTarget(e.target.value)}
                  className="px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-200 outline-none"
                >
                  <option value="success_rate">Success Rate</option>
                  <option value="fraud_risk">Fraud Risk</option>
                  <option value="expected_amount">Expected Amount</option>
                </select>
              </div>
              <p className="text-xs text-slate-400 mb-4">Shows how much each parameter can swing the predicted outcome (wider = more impact)</p>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={tornadoData} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                    <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#475569' }} width={100} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                    <Bar dataKey="low" stackId="a" fill="#f87171" radius={[4, 0, 0, 4]} name="Negative Impact" />
                    <Bar dataKey="high" stackId="a" fill="#34d399" radius={[0, 4, 4, 0]} name="Positive Impact" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Overrides Summary */}
            {Object.keys(result.overrides_applied).length > 0 && (
              <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-4">
                <h4 className="text-sm font-semibold text-indigo-800 mb-2">Parameters Changed</h4>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(result.overrides_applied).map(([k, v]) => (
                    <span key={k} className="inline-flex items-center gap-1 px-3 py-1 bg-white border border-indigo-200 rounded-full text-xs font-medium text-indigo-700">
                      {prettyLabel(k)}: <strong>{String(v)}</strong>
                    </span>
                  ))}
                </div>
              </div>
            )}
            {/* AI Recommendations */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-yellow-100 flex items-center justify-center">
                    <Lightbulb className="w-4 h-4 text-yellow-600" />
                  </div>
                  <h3 className="font-semibold text-slate-800">AI Recommendations</h3>
                </div>
                <div className="flex items-center gap-2">
                  <select value={recGoal} onChange={e => setRecGoal(e.target.value)} className="px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-sm outline-none">
                    <option value="maximize">Maximize</option>
                    <option value="minimize">Minimize</option>
                  </select>
                  <select value={recTarget} onChange={e => setRecTarget(e.target.value)} className="px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-sm outline-none">
                    <option value="success_rate">Success Rate</option>
                    <option value="fraud_risk">Fraud Risk</option>
                    <option value="expected_amount">Expected Amount</option>
                  </select>
                  <button onClick={fetchRecommendations} disabled={recLoading} className="px-4 py-1.5 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 flex items-center gap-2">
                    {recLoading ? <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" /> : <Sparkles className="w-3 h-3" />}
                    Get Ideas
                  </button>
                </div>
              </div>

              {recResult ? (
                <div className="space-y-4">
                  <p className="text-sm text-indigo-700 bg-indigo-50 p-3 rounded-lg font-medium">{recResult.explanation}</p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {recResult.best.map((b, i) => (
                      <div key={i} className="border border-emerald-200 bg-emerald-50/30 p-4 rounded-xl flex flex-col justify-between">
                        <div>
                          <div className="text-xs font-bold text-emerald-600 uppercase tracking-wide mb-1">Option {i+1}</div>
                          <div className="text-2xl font-bold text-slate-800 mb-2">{b.predicted}{recTarget === 'expected_amount' ? '' : '%'}</div>
                          <div className="text-xs text-slate-600 leading-relaxed mb-4">{b.summary}</div>
                        </div>
                        <button onClick={() => applyRecommendation(b.overrides)} className="w-full py-1.5 bg-emerald-600 text-white rounded-lg text-xs font-medium hover:bg-emerald-700 transition-colors flex items-center justify-center gap-1">
                          Apply <ArrowRight className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-6 text-sm text-slate-400 border-2 border-dashed border-slate-200 rounded-xl">
                  Click "Get Ideas" to automatically find the best parameter combinations
                </div>
              )}
            </div>

            {/* Saved Scenarios Comparison */}
            {savedScenarios.length > 0 && (
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="flex items-center gap-2 mb-4">
                  <Layers className="w-5 h-5 text-indigo-600" />
                  <h3 className="font-semibold text-slate-800">Scenario Comparison</h3>
                </div>
                
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-slate-50 text-slate-600 font-medium">
                      <tr>
                        <th className="px-4 py-3 rounded-tl-lg">Metric</th>
                        <th className="px-4 py-3">Baseline</th>
                        {savedScenarios.map((s, i) => (
                          <th key={i} className="px-4 py-3">
                            <div className="flex items-center justify-between">
                              {s.name}
                              <button onClick={() => deleteScenario(i)} className="text-slate-400 hover:text-red-500"><X className="w-3 h-3" /></button>
                            </div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      <tr>
                        <td className="px-4 py-3 font-medium text-slate-700">Success Rate</td>
                        <td className="px-4 py-3 text-slate-500">{result?.success_rate.baseline}%</td>
                        {savedScenarios.map((s, i) => (
                          <td key={i} className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              {s.result.success_rate.simulated}% <DeltaChip value={s.result.success_rate.delta} />
                            </div>
                          </td>
                        ))}
                      </tr>
                      <tr>
                        <td className="px-4 py-3 font-medium text-slate-700">Fraud Risk</td>
                        <td className="px-4 py-3 text-slate-500">{result?.fraud_risk.baseline}%</td>
                        {savedScenarios.map((s, i) => (
                          <td key={i} className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              {s.result.fraud_risk.simulated}% <DeltaChip value={s.result.fraud_risk.delta} invert />
                            </div>
                          </td>
                        ))}
                      </tr>
                      <tr>
                        <td className="px-4 py-3 font-medium text-slate-700">Amount (INR)</td>
                        <td className="px-4 py-3 text-slate-500">₹{result?.expected_amount.baseline}</td>
                        {savedScenarios.map((s, i) => (
                          <td key={i} className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              ₹{s.result.expected_amount.simulated} <DeltaChip value={s.result.expected_amount.delta} />
                            </div>
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
