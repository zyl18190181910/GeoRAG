from __future__ import annotations


HOME_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GeoRAG · 地学资料智能问答</title>
  <style>
    :root {
      color-scheme: light;
      --navy:#10251f; --ink:#17231f; --muted:#687670; --brand:#2f765d; --brand-dark:#245d4a;
      --brand-soft:#e8f2ee; --sand:#e7c89c; --canvas:#f4f6f4; --surface:#fff; --line:#dfe6e2;
      --line-strong:#c9d5cf; --danger:#a94436;
    }
    * { box-sizing:border-box; }
    html, body { height:100%; }
    body { margin:0; color:var(--ink); background:var(--canvas); font:15px/1.65 Inter,"Microsoft YaHei",system-ui,sans-serif; }
    button, input, textarea, select { font:inherit; }
    button { cursor:pointer; }
    button:disabled { opacity:.55; cursor:wait; }
    .app { min-height:100vh; display:grid; grid-template-columns:276px minmax(0,1fr); }
    .sidebar { position:sticky; top:0; height:100vh; display:flex; flex-direction:column; color:#e9f0ed; background:var(--navy); padding:26px 18px 18px; border-right:1px solid rgba(255,255,255,.06); }
    .brand { display:flex; align-items:center; gap:12px; padding:0 8px 26px; }
    .brand-mark { width:42px; height:42px; display:grid; place-items:center; border-radius:13px; color:#17352b; background:var(--sand); font-weight:900; letter-spacing:-1px; box-shadow:inset 0 0 0 1px rgba(255,255,255,.35); }
    .brand strong { display:block; color:#fff; font-size:17px; letter-spacing:.2px; }
    .brand span { color:#9cb0a9; font-size:12px; }
    .nav-label { padding:0 10px 8px; color:#779087; font-size:11px; font-weight:800; letter-spacing:1.6px; }
    .nav { display:grid; gap:7px; }
    .nav-button { width:100%; display:flex; align-items:center; gap:11px; border:0; border-radius:11px; padding:11px 12px; color:#b8c9c3; background:transparent; text-align:left; font-weight:650; }
    .nav-button:hover { color:#fff; background:rgba(255,255,255,.06); }
    .nav-button.active { color:#fff; background:rgba(231,200,156,.14); box-shadow:inset 3px 0 0 var(--sand); }
    .nav-icon { width:23px; color:var(--sand); font-size:18px; text-align:center; }
    .sidebar-card { margin-top:24px; padding:15px; border:1px solid rgba(255,255,255,.08); border-radius:14px; background:rgba(255,255,255,.035); }
    .sidebar-card strong { display:block; color:#fff; margin-bottom:7px; }
    .mini-stat { display:flex; justify-content:space-between; color:#9fb2ab; font-size:13px; margin-top:5px; }
    .mini-stat b { color:var(--sand); }
    .sidebar-footer { margin-top:auto; padding:18px 8px 2px; color:#7e978e; font-size:12px; }
    .sidebar-footer a { color:#b9cbc5; }
    .workspace { min-width:0; height:100vh; overflow:hidden; display:flex; flex-direction:column; }
    .topbar { height:72px; flex:0 0 72px; display:flex; align-items:center; justify-content:space-between; gap:18px; padding:0 30px; background:rgba(255,255,255,.92); border-bottom:1px solid var(--line); backdrop-filter:blur(14px); }
    .topbar h1 { margin:0; font-size:18px; letter-spacing:-.2px; }
    .topbar p { margin:1px 0 0; color:var(--muted); font-size:12px; }
    .status-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
    .status-pill { display:inline-flex; align-items:center; gap:6px; padding:5px 9px; border:1px solid var(--line); border-radius:999px; color:#52635d; background:#fafcfb; font-size:12px; white-space:nowrap; }
    .status-pill::before { content:""; width:6px; height:6px; border-radius:50%; background:#75a998; box-shadow:0 0 0 3px #e2f1eb; }
    .panel { display:none; min-height:0; flex:1; overflow:auto; }
    .panel.active { display:flex; }
    #chatPanel { flex-direction:column; }
    .chat-scroll { flex:1; overflow:auto; scroll-behavior:smooth; }
    .chat-content { width:min(920px,calc(100% - 40px)); margin:0 auto; padding:50px 0 28px; }
    .welcome { padding:10px 0 34px; }
    .eyebrow { margin:0 0 8px; color:var(--brand); font-size:12px; font-weight:850; letter-spacing:1.8px; text-transform:uppercase; }
    .welcome h2 { max-width:700px; margin:0; font-size:clamp(30px,4vw,48px); line-height:1.18; letter-spacing:-1.6px; }
    .welcome h2 span { color:var(--brand); }
    .welcome-copy { max-width:680px; margin:14px 0 20px; color:var(--muted); font-size:16px; }
    .tech-line { display:flex; flex-wrap:wrap; align-items:center; gap:7px; }
    .tech-node { padding:6px 10px; border:1px solid var(--line); border-radius:8px; color:#42534d; background:#fff; font-size:12px; font-weight:700; }
    .tech-arrow { color:#9aa8a3; font-size:13px; }
    .starters { display:grid; grid-template-columns:repeat(3,1fr); gap:11px; margin-top:28px; }
    .starter { min-height:104px; padding:15px; border:1px solid var(--line); border-radius:14px; color:var(--ink); background:rgba(255,255,255,.76); text-align:left; transition:.18s ease; }
    .starter:hover { border-color:#aebfb7; background:#fff; transform:translateY(-2px); box-shadow:0 10px 24px rgba(16,37,31,.06); }
    .starter small { display:block; color:var(--brand); font-weight:800; margin-bottom:6px; }
    .starter span { color:#475852; font-size:13px; line-height:1.5; }
    .message-row { display:grid; grid-template-columns:38px minmax(0,1fr); gap:13px; margin:26px 0; }
    .avatar { width:38px; height:38px; display:grid; place-items:center; border-radius:12px; color:#fff; background:var(--brand); font-size:12px; font-weight:850; }
    .message-row.user .avatar { color:#34443e; background:#e2e7e4; }
    .message-head { display:flex; align-items:center; justify-content:space-between; gap:10px; min-height:30px; }
    .message-head strong { font-size:14px; }
    .message-body { margin-top:4px; font-size:15px; }
    .message-row.user .message-body { white-space:pre-wrap; }
    .answer-copy { border:1px solid var(--line); border-radius:8px; padding:5px 9px; color:var(--muted); background:#fff; font-size:12px; }
    .answer-copy:hover { color:var(--brand); border-color:#b5c7bf; }
    .answer-meta { margin:12px 0 0; color:var(--muted); font-size:12px; }
    .answer-warning { margin-top:12px; padding:9px 11px; border-radius:9px; color:#795f22; background:#fff7df; font-size:12px; }
    .answer-table { width:100%; border-collapse:separate; border-spacing:0; margin:13px 0; font-size:13px; overflow:hidden; border:1px solid var(--line); border-radius:10px; }
    .answer-table th, .answer-table td { padding:10px 11px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    .answer-table th:last-child, .answer-table td:last-child { border-right:0; }
    .answer-table tr:last-child td { border-bottom:0; }
    .answer-table th { color:#28483d; background:var(--brand-soft); }
    .answer-paragraph { margin:8px 0; white-space:pre-wrap; }
    .answer-heading { margin:14px 0 6px; font-size:16px; }
    .answer-list { margin:7px 0; padding-left:21px; }
    .answer-list li { margin:4px 0; }
    .trace { margin-top:15px; border:1px solid var(--line); border-radius:12px; background:#fbfcfb; overflow:hidden; }
    .trace summary { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:11px 13px; cursor:pointer; list-style:none; color:#42544d; font-size:13px; font-weight:750; }
    .trace summary::-webkit-details-marker { display:none; }
    .trace summary::after { content:"＋"; color:var(--brand); }
    .trace[open] summary::after { content:"－"; }
    .trace-content { padding:0 13px 13px; border-top:1px solid var(--line); }
    .trace-label { margin:12px 0 5px; color:#7a8984; font-size:11px; font-weight:850; letter-spacing:1px; }
    .query-chip { display:inline-block; margin:2px 5px 2px 0; padding:4px 8px; border-radius:7px; color:#3b554b; background:var(--brand-soft); font-size:12px; }
    .citation { margin-top:8px; padding:11px 12px; border-left:3px solid #78a793; border-radius:8px; background:#f3f7f5; }
    .citation strong { display:block; color:#2f4b40; font-size:12px; }
    .citation span { display:block; margin-top:3px; color:#55665f; font-size:12px; }
    .loading-card { padding:14px; border:1px solid var(--line); border-radius:13px; background:#fff; }
    .loading-title { display:flex; align-items:center; gap:9px; color:#334740; font-weight:750; }
    .spinner { width:15px; height:15px; border:2px solid #cbd8d2; border-top-color:var(--brand); border-radius:50%; animation:spin .8s linear infinite; }
    .pipeline { display:flex; flex-wrap:wrap; gap:6px; margin-top:11px; }
    .pipeline-step { padding:5px 8px; border-radius:7px; color:#8a9691; background:#f0f3f1; font-size:11px; }
    .pipeline-step.done { color:#31624f; background:#e6f1ec; }
    .pipeline-step.active { color:#fff; background:var(--brand); }
    @keyframes spin { to { transform:rotate(360deg); } }
    .composer-wrap { flex:0 0 auto; padding:12px 24px 18px; background:linear-gradient(180deg,rgba(244,246,244,0),var(--canvas) 24%); }
    .composer { width:min(920px,100%); margin:0 auto; padding:10px 11px 10px 15px; border:1px solid var(--line-strong); border-radius:17px; background:#fff; box-shadow:0 12px 35px rgba(16,37,31,.09); }
    .composer textarea { display:block; width:100%; min-height:50px; max-height:150px; resize:none; border:0; outline:0; padding:7px 5px; color:var(--ink); background:transparent; }
    .composer textarea::placeholder { color:#99a49f; }
    .composer-tools { display:flex; align-items:center; justify-content:space-between; gap:10px; padding-top:7px; }
    .composer-left { display:flex; align-items:center; gap:8px; min-width:0; }
    .mode-select { max-width:210px; border:1px solid var(--line); border-radius:9px; padding:7px 9px; color:#41534c; background:#f8faf9; font-size:12px; }
    .clear-button { border:0; padding:6px 8px; color:#7a8883; background:transparent; font-size:12px; }
    .send-button { width:40px; height:40px; border:0; border-radius:12px; color:#fff; background:var(--brand); font-size:20px; line-height:1; }
    .send-button:hover { background:var(--brand-dark); }
    .composer-note { margin:7px 0 0; color:#899690; text-align:center; font-size:11px; }
    .page { width:min(1080px,calc(100% - 48px)); margin:0 auto; padding:38px 0 70px; }
    .page-heading { display:flex; justify-content:space-between; align-items:flex-start; gap:22px; margin-bottom:24px; }
    .page-heading h2 { margin:0; font-size:28px; letter-spacing:-.7px; }
    .page-heading p { margin:5px 0 0; color:var(--muted); }
    .card { padding:22px; border:1px solid var(--line); border-radius:16px; background:#fff; box-shadow:0 10px 30px rgba(16,37,31,.045); }
    .card h3 { margin:0 0 5px; font-size:17px; }
    .card p { margin:4px 0; }
    .upload-grid { display:grid; grid-template-columns:1.05fr .95fr; gap:16px; }
    .dropzone { display:block; margin-top:16px; padding:28px; border:1px dashed #acc0b7; border-radius:13px; background:#f8faf9; text-align:center; cursor:pointer; }
    .dropzone:hover { border-color:var(--brand); background:#f1f7f4; }
    .dropzone strong { display:block; color:#2d493e; }
    .dropzone span { color:var(--muted); font-size:12px; }
    input[type=file] { position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }
    .primary, .secondary, .danger { border:0; border-radius:10px; padding:9px 14px; font-weight:750; }
    .primary { color:#fff; background:var(--brand); }
    .secondary { color:#285442; background:var(--brand-soft); }
    .danger { color:var(--danger); background:#fff0ed; }
    .small { padding:6px 9px; font-size:12px; }
    .action-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:13px; }
    .message { min-height:22px; margin-top:10px; color:var(--muted); font-size:13px; }
    .error { color:#a12b2b; }
    .privacy { padding:16px; border-radius:12px; color:#665736; background:#fbf5e9; font-size:13px; }
    .privacy strong { display:block; color:#56471f; margin-bottom:5px; }
    .source-section { margin-top:16px; }
    .section-bar { display:flex; justify-content:space-between; align-items:center; gap:14px; margin-bottom:12px; }
    .section-bar h3 { margin:0; }
    .source-list { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .source-row { display:flex; align-items:center; gap:10px; padding:14px; border:1px solid var(--line); border-radius:12px; background:#fbfcfb; }
    .source-file-icon { width:35px; height:35px; flex:0 0 35px; display:grid; place-items:center; border-radius:9px; color:#356550; background:var(--brand-soft); font-size:11px; font-weight:850; }
    .source-main { min-width:0; flex:1; }
    .source-name { display:block; overflow:hidden; color:#2d4039; font-size:13px; font-weight:750; text-overflow:ellipsis; white-space:nowrap; }
    .source-meta { color:var(--muted); font-size:11px; }
    .metrics { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:18px 0; }
    .metric { padding:17px 12px; border:1px solid var(--line); border-radius:13px; background:#fff; }
    .metric strong { display:block; color:var(--brand-dark); font-size:25px; line-height:1.2; }
    .metric span { color:var(--muted); font-size:11px; }
    .architecture { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin:18px 0; }
    .architecture div { position:relative; padding:15px 8px; border:1px solid var(--line); border-radius:11px; background:#f8faf9; text-align:center; }
    .architecture div:not(:last-child)::after { content:"→"; position:absolute; right:-10px; top:50%; z-index:2; transform:translateY(-50%); color:#90a099; }
    .architecture strong { display:block; color:#315144; font-size:12px; }
    .architecture span { color:#7b8984; font-size:10px; }
    .evaluation-detail { margin-top:8px; padding:11px 12px; border-left:3px solid #79a995; border-radius:8px; color:#4b5e56; background:#f4f8f6; font-size:12px; }
    .evaluation-layout { display:grid; grid-template-columns:1.35fr .65fr; gap:16px; }
    .eval-actions { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
    .ablation-head { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; }
    .ablation-summary { flex:0 0 auto; padding:8px 11px; border-radius:9px; color:#315a4a; background:var(--brand-soft); font-size:12px; font-weight:750; }
    .method-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:17px; }
    .method-card { padding:15px; border:1px solid var(--line); border-radius:12px; background:#fafcfb; }
    .method-card.featured { border-color:#8fb5a5; background:#eef6f2; box-shadow:inset 0 3px 0 var(--brand); }
    .method-card small { display:block; color:var(--muted); font-size:10px; }
    .method-card h4 { margin:3px 0 11px; font-size:14px; }
    .method-score { display:flex; align-items:flex-end; justify-content:space-between; gap:8px; }
    .method-score strong { color:var(--brand-dark); font-size:26px; line-height:1; }
    .method-score span { color:var(--muted); font-size:10px; }
    .metric-bars { display:grid; gap:8px; margin-top:13px; }
    .bar-row { display:grid; grid-template-columns:52px minmax(70px,1fr) 40px; gap:8px; align-items:center; font-size:10px; color:#697771; }
    .bar-track { height:6px; overflow:hidden; border-radius:999px; background:#dfe8e4; }
    .bar-fill { height:100%; border-radius:999px; background:var(--brand); }
    .ablation-table { width:100%; border-collapse:collapse; margin-top:17px; font-size:12px; }
    .ablation-table th, .ablation-table td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; }
    .ablation-table th:first-child, .ablation-table td:first-child { text-align:left; }
    .ablation-table th { color:#64736d; font-size:10px; letter-spacing:.5px; text-transform:uppercase; }
    .ablation-table tr:last-child td { border-bottom:0; }
    .ablation-footnote { margin:12px 0 0; color:#83908b; font-size:11px; }
    .showcase-note { color:var(--muted); font-size:13px; }
    a { color:var(--brand); }
    @media (max-width:900px) {
      .app { grid-template-columns:84px minmax(0,1fr); } .sidebar { padding:22px 12px; } .brand { justify-content:center; padding:0 0 25px; }
      .brand-copy, .nav-label, .nav-button span:not(.nav-icon), .sidebar-card, .sidebar-footer { display:none; }
      .nav-button { justify-content:center; padding:11px; } .nav-icon { width:auto; } .topbar { padding:0 20px; }
      .status-row .status-pill:nth-child(n+3) { display:none; } .upload-grid, .evaluation-layout { grid-template-columns:1fr; } .metrics { grid-template-columns:repeat(3,1fr); } .method-grid { grid-template-columns:1fr; }
    }
    @media (max-width:620px) {
      .app { display:block; } .sidebar { position:fixed; z-index:20; top:auto; bottom:0; width:100%; height:64px; padding:7px 10px; }
      .brand, .nav-label, .sidebar-card, .sidebar-footer { display:none; } .nav { grid-template-columns:repeat(3,1fr); }
      .nav-button { display:grid; gap:0; justify-items:center; padding:4px; font-size:10px; } .nav-button span:not(.nav-icon) { display:block; } .nav-icon { font-size:16px; }
      .workspace { height:calc(100vh - 64px); } .topbar { height:62px; flex-basis:62px; padding:0 15px; } .topbar p, .status-row .status-pill:not(:first-child) { display:none; }
      .chat-content { width:min(100% - 26px,920px); padding-top:28px; } .welcome h2 { font-size:31px; } .starters { grid-template-columns:1fr; } .starter { min-height:auto; }
      .tech-line { display:none; } .composer-wrap { padding:9px 12px 12px; } .mode-select { max-width:170px; }
      .page { width:calc(100% - 26px); padding-top:24px; } .page-heading { display:block; } .source-list, .metrics { grid-template-columns:1fr; } .architecture { grid-template-columns:1fr; }
      .architecture div:not(:last-child)::after { content:"↓"; right:auto; left:50%; top:auto; bottom:-14px; transform:translateX(-50%); }
    }
  </style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand"><div class="brand-mark">G</div><div class="brand-copy"><strong>GeoRAG</strong><span>地学知识工作台</span></div></div>
    <div class="nav-label">WORKSPACE</div>
    <nav class="nav" aria-label="主导航">
      <button class="nav-button active" data-panel="chatPanel" data-title="智能问答" data-subtitle="基于资料回答，每条结论都可追溯"><span class="nav-icon">✦</span><span>智能问答</span></button>
      <button class="nav-button" data-panel="libraryPanel" data-title="资料库" data-subtitle="管理检索范围和知识库索引"><span class="nav-icon">▤</span><span>资料库</span></button>
      <button class="nav-button" data-panel="evaluationPanel" data-title="效果评测" data-subtitle="用可复现指标验证系统效果"><span class="nav-icon">◫</span><span>效果评测</span></button>
    </nav>
    <div class="sidebar-card"><strong>知识库状态</strong><div class="mini-stat"><span>启用资料</span><b id="sideSourceCount">0</b></div><div class="mini-stat"><span>有效片段</span><b id="sideChunkCount">0</b></div><div class="mini-stat"><span>检索链路</span><b>Hybrid RAG</b></div></div>
    <div class="sidebar-footer">作品集演示项目<br><a href="/docs">查看 API 文档 ↗</a></div>
  </aside>
  <main class="workspace">
    <header class="topbar">
      <div><h1 id="pageTitle">智能问答</h1><p id="pageSubtitle">基于资料回答，每条结论都可追溯</p></div>
      <div class="status-row"><span class="status-pill" id="serviceStatus">连接中</span><span class="status-pill" id="modelStatus">检测模型</span><span class="status-pill" id="rerankerStatus">检测重排</span></div>
    </header>
    <section class="panel active" id="chatPanel">
      <div class="chat-scroll" id="chatScroll"><div class="chat-content">
        <div class="welcome" id="welcome">
          <p class="eyebrow">Retrieval · Reasoning · Evidence</p><h2>让地学资料里的每个答案<br><span>都有证据可循</span></h2>
          <p class="welcome-copy">面向课程资料、项目报告与公开论文的可追溯问答系统。支持事实查找、跨文档归纳和有依据的分析。</p>
          <div class="tech-line" aria-label="系统处理流程"><span class="tech-node">问题理解</span><span class="tech-arrow">→</span><span class="tech-node">混合检索</span><span class="tech-arrow">→</span><span class="tech-node">BGE 重排</span><span class="tech-arrow">→</span><span class="tech-node">DeepSeek 生成</span><span class="tech-arrow">→</span><span class="tech-node">引用校验</span></div>
          <div class="starters">
            <button class="starter" data-mode="summary" data-question="请归纳两篇遥感语义分割论文分别解决了什么问题、使用了什么方法，并整理成表格。"><small>跨资料归纳</small><span>对比两篇遥感语义分割论文的方法与问题</span></button>
            <button class="starter" data-mode="analysis" data-question="根据资料，分析地学先验知识对遥感语义分割结果有什么帮助，并说明资料尚未覆盖的内容。"><small>分析建议</small><span>分析地学先验知识对遥感语义分割的帮助</span></button>
            <button class="starter" data-mode="fact" data-question="Transformer 大地电磁反演使用了多少个浅部和深部异常体模型？"><small>事实查找</small><span>查找 Transformer 反演的训练样本数量</span></button>
          </div>
        </div>
        <div id="chatMessages" aria-live="polite"></div>
      </div></div>
      <div class="composer-wrap"><div class="composer">
        <textarea id="question" rows="1" placeholder="输入关于资料的问题，Enter 发送，Shift + Enter 换行"></textarea>
        <div class="composer-tools"><div class="composer-left"><select class="mode-select" id="answerMode" aria-label="回答方式"><option value="auto">智能判断</option><option value="fact">事实查找</option><option value="summary">归纳总结</option><option value="analysis">分析建议</option></select><button class="clear-button" id="clearButton" type="button">清空对话</button></div><button class="send-button" id="askButton" type="button" aria-label="发送问题">↑</button></div>
      </div><p class="composer-note" id="askMessage">回答仅依据已启用资料；智能模式会将相关片段发送给 DeepSeek。</p></div>
    </section>
    <section class="panel" id="libraryPanel"><div class="page">
      <div class="page-heading"><div><p class="eyebrow">KNOWLEDGE BASE</p><h2>资料库</h2><p>控制哪些资料可以参与检索，并维护本地索引。</p></div><button id="reindexButton" class="secondary" type="button">重新建立索引</button></div>
      <div class="upload-grid">
        <div class="card"><h3>添加资料</h3><p class="showcase-note">支持 PDF、TXT 和 Markdown，单个文件不超过 10 MB。</p><label class="dropzone" for="files"><strong>点击选择文件</strong><span id="fileHint">可同时上传多份资料</span></label><input id="files" type="file" accept=".pdf,.txt,.md" multiple><div class="action-row"><button id="uploadButton" class="primary" type="button">上传并建立索引</button></div><div id="uploadMessage" class="message" aria-live="polite"></div></div>
        <div class="privacy"><strong>数据与隐私说明</strong>智能生成只发送与问题最相关的少量片段。请勿导入保密材料；受版权保护的书籍只用于本机测试，不要提交到公开仓库。扫描图片版 PDF 需要先完成 OCR。</div>
      </div>
      <div class="card source-section"><div class="section-bar"><div><h3>当前资料</h3><p class="showcase-note" id="sourceCount">正在读取资料列表…</p></div></div><div id="sourceList" class="source-list"><span class="showcase-note">正在读取…</span></div></div>
    </div></section>
    <section class="panel" id="evaluationPanel"><div class="page">
      <div class="page-heading"><div><p class="eyebrow">EVALUATION</p><h2>效果评测</h2><p>用可复现的公开资料验证生成质量和引用有效性。</p></div><div class="eval-actions"><button id="evaluationButton" class="primary" type="button">运行生成评测</button></div></div>
      <div class="metrics"><div class="metric"><strong id="metricPass">—</strong><span>通过用例</span></div><div class="metric"><strong id="metricGeneration">—</strong><span>生成成功率</span></div><div class="metric"><strong id="metricCoverage">—</strong><span>来源覆盖率</span></div><div class="metric"><strong id="metricRecall">—</strong><span>要点召回率</span></div><div class="metric"><strong id="metricCitation">—</strong><span>引用有效率</span></div></div>
      <div class="evaluation-layout">
        <div class="card"><h3>系统处理链路</h3><p class="showcase-note">从问题到带引用回答的完整 RAG 流程。</p><div class="architecture"><div><strong>Query Rewrite</strong><span>查询改写</span></div><div><strong>Hybrid Search</strong><span>关键词 + 向量</span></div><div><strong>BGE Reranker</strong><span>交叉编码器精排</span></div><div><strong>DeepSeek</strong><span>结构化生成</span></div><div><strong>Citation Check</strong><span>引用有效性校验</span></div></div></div>
        <div class="card"><h3>评测说明</h3><p class="showcase-note">生成评测使用项目原创样例，不会读取用户上传的私人文件；结果仅反映当前数据集与配置。</p></div>
      </div>
      <div class="card source-section"><div id="evaluationMessage" class="message" aria-live="polite">正在读取最近报告…</div><div id="evaluationDetails"></div></div>
    </div></section>
  </main>
</div>
<script>
  const byId = id => document.getElementById(id);
  const conversation = [];
  let stageTimer = null;
  async function readError(response) { try { const value = await response.json(); return value.detail || JSON.stringify(value); } catch { return `请求失败（HTTP ${response.status}）`; } }
  document.querySelectorAll('.nav-button').forEach(button => { button.addEventListener('click', () => { document.querySelectorAll('.nav-button').forEach(item => item.classList.toggle('active', item === button)); document.querySelectorAll('.panel').forEach(panel => panel.classList.toggle('active', panel.id === button.dataset.panel)); byId('pageTitle').textContent = button.dataset.title; byId('pageSubtitle').textContent = button.dataset.subtitle; }); });
  function tableCells(line) { return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(value => value.trim()); }
  function isTableDivider(line) { const cells = tableCells(line); return cells.length > 1 && cells.every(value => /^:?-{3,}:?$/.test(value)); }
  function prepareAnswerDisplay(text, citations) {
    let cleaned = String(text || '').replace(/\*\*/g, '');
    const numberedCitations = (citations || []).map((item, index) => ({...item, displayIndex:index + 1}));
    numberedCitations.forEach(item => {
      cleaned = cleaned.split(`[${item.chunk_id}]`).join(`[${item.displayIndex}]`);
    });
    const firstNumberBySource = new Map();
    numberedCitations.forEach(item => {
      if (!firstNumberBySource.has(item.source_id)) firstNumberBySource.set(item.source_id, item.displayIndex);
    });
    firstNumberBySource.forEach((displayIndex, sourceId) => {
      cleaned = cleaned.split(`[${sourceId}]`).join(`[${displayIndex}]`);
    });
    return {text:cleaned, citations:numberedCitations};
  }
  function citationSourceTitle(item) {
    return String(item.filename || item.source_id || '资料来源').replace(/^\d+[_-]/, '').replace(/\.[^.]+$/, '');
  }
  function renderAnswer(text, root) {
    root.replaceChildren(); const lines = text.split('\n'); let paragraph = []; let list = null;
    const flushParagraph = () => { if (!paragraph.length) return; const block = document.createElement('p'); block.className = 'answer-paragraph'; block.textContent = paragraph.join('\n'); root.append(block); paragraph = []; };
    const flushList = () => { if (list) { root.append(list); list = null; } };
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      if (index + 1 < lines.length && line.includes('|') && isTableDivider(lines[index + 1])) {
        flushParagraph(); flushList(); const table = document.createElement('table'); table.className = 'answer-table'; const head = document.createElement('thead'); const headRow = document.createElement('tr');
        tableCells(line).forEach(value => { const cell = document.createElement('th'); cell.textContent = value; headRow.append(cell); }); head.append(headRow); table.append(head); const body = document.createElement('tbody'); index += 2;
        while (index < lines.length && lines[index].includes('|') && lines[index].trim()) { const row = document.createElement('tr'); tableCells(lines[index]).forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }); body.append(row); index += 1; }
        table.append(body); root.append(table); index -= 1; continue;
      }
      const heading = line.match(/^#{1,3}\s+(.+)/); if (heading) { flushParagraph(); flushList(); const title = document.createElement('h3'); title.className = 'answer-heading'; title.textContent = heading[1]; root.append(title); continue; }
      const bullet = line.match(/^\s*[-*]\s+(.+)/); if (bullet) { flushParagraph(); if (!list) { list = document.createElement('ul'); list.className = 'answer-list'; } const item = document.createElement('li'); item.textContent = bullet[1]; list.append(item); continue; }
      if (!line.trim()) { flushParagraph(); flushList(); continue; } flushList(); paragraph.push(line);
    }
    flushParagraph(); flushList();
  }
  function scrollChat() { requestAnimationFrame(() => { const area = byId('chatScroll'); area.scrollTop = area.scrollHeight; }); }
  function addUserMessage(text) { const row = document.createElement('article'); row.className = 'message-row user'; row.innerHTML = '<div class="avatar">你</div><div><div class="message-head"><strong>你的问题</strong></div><div class="message-body"></div></div>'; row.querySelector('.message-body').textContent = text; byId('chatMessages').append(row); scrollChat(); }
  function addLoadingMessage() {
    const row = document.createElement('article'); row.className = 'message-row assistant'; const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = 'AI'; const body = document.createElement('div'); const card = document.createElement('div'); card.className = 'loading-card'; card.innerHTML = '<div class="loading-title"><span class="spinner"></span><span>正在查找可靠答案</span></div><div class="pipeline"></div>';
    const labels = ['问题理解','混合检索','BGE 重排','DeepSeek 生成','引用校验']; labels.forEach((label, index) => { const step = document.createElement('span'); step.className = `pipeline-step${index === 0 ? ' active' : ''}`; step.textContent = label; card.querySelector('.pipeline').append(step); }); body.append(card); row.append(avatar, body); byId('chatMessages').append(row); scrollChat();
    let current = 0; clearInterval(stageTimer); stageTimer = setInterval(() => { const steps = row.querySelectorAll('.pipeline-step'); if (current < steps.length) steps[current].className = 'pipeline-step done'; current += 1; if (current < steps.length) steps[current].className = 'pipeline-step active'; }, 1200); return row;
  }
  function renderAssistantMessage(row, value) {
    clearInterval(stageTimer); const display = prepareAnswerDisplay(value.answer, value.citations); const content = row.children[1]; content.replaceChildren(); const head = document.createElement('div'); head.className = 'message-head'; const title = document.createElement('strong'); title.textContent = value.refused ? '未找到充分依据' : 'GeoRAG'; const copy = document.createElement('button'); copy.className = 'answer-copy'; copy.textContent = '复制回答';
    copy.addEventListener('click', async () => { await navigator.clipboard.writeText(display.text); copy.textContent = '已复制'; setTimeout(() => { copy.textContent = '复制回答'; }, 1500); }); head.append(title, copy);
    const answer = document.createElement('div'); answer.className = 'message-body'; renderAnswer(display.text, answer); const labels = {fact:'事实查找', summary:'归纳总结', analysis:'分析建议'}; const meta = document.createElement('p'); meta.className = 'answer-meta'; meta.textContent = `${value.mode === 'deepseek' ? 'DeepSeek 生成' : '本地抽取'} · ${labels[value.answer_mode] || value.answer_mode} · ${(value.latency_ms / 1000).toFixed(2)} 秒 · ${display.citations.length} 条引用`; content.append(head, answer, meta);
    if (value.warning) { const warning = document.createElement('div'); warning.className = 'answer-warning'; warning.textContent = value.warning; content.append(warning); }
    const trace = document.createElement('details'); trace.className = 'trace'; const summary = document.createElement('summary'); summary.innerHTML = '<span>查看检索过程与引用证据</span>'; const traceContent = document.createElement('div'); traceContent.className = 'trace-content'; const queryLabel = document.createElement('div'); queryLabel.className = 'trace-label'; queryLabel.textContent = '查询规划'; traceContent.append(queryLabel);
    (value.rewritten_queries.length ? value.rewritten_queries : ['使用原始问题检索']).forEach(query => { const chip = document.createElement('span'); chip.className = 'query-chip'; chip.textContent = query; traceContent.append(chip); }); const sourceLabel = document.createElement('div'); sourceLabel.className = 'trace-label'; sourceLabel.textContent = '引用证据'; traceContent.append(sourceLabel);
    if (!display.citations.length) { const none = document.createElement('span'); none.className = 'answer-meta'; none.textContent = '没有可展示的引用。'; traceContent.append(none); }
    display.citations.forEach(item => { const card = document.createElement('div'); card.className = 'citation'; const source = document.createElement('strong'); const location = item.page ? `第 ${item.page} 页` : item.section ? `章节：${item.section}` : '文档片段'; source.textContent = `[${item.displayIndex}] ${citationSourceTitle(item)} · ${location} · 相关度 ${item.score.toFixed(3)}`; const excerpt = document.createElement('span'); excerpt.textContent = item.excerpt; card.append(source, excerpt); traceContent.append(card); }); trace.append(summary, traceContent); content.append(trace); scrollChat();
  }
  async function refreshStatus() {
    try {
      const [healthResponse, sourcesResponse] = await Promise.all([fetch('/health'), fetch('/sources/details')]); const health = await healthResponse.json(); const sources = await sourcesResponse.json(); byId('serviceStatus').textContent = '服务正常'; byId('modelStatus').textContent = health.semantic_enabled && health.generation_enabled ? health.generation_model : '基础检索'; byId('rerankerStatus').textContent = health.reranker_enabled ? 'BGE 精排已启用' : '精排未启用'; byId('sideSourceCount').textContent = `${health.active_sources}/${health.sources}`; byId('sideChunkCount').textContent = health.active_chunks; byId('sourceCount').textContent = `共 ${health.sources} 份资料，${health.active_sources} 份参与检索`;
      const list = byId('sourceList'); list.replaceChildren(); if (!sources.length) { const item = document.createElement('span'); item.className = 'showcase-note'; item.textContent = '还没有资料，请先上传。'; list.append(item); }
      sources.forEach(source => { const row = document.createElement('div'); row.className = 'source-row'; const icon = document.createElement('div'); icon.className = 'source-file-icon'; icon.textContent = 'DOC'; const main = document.createElement('div'); main.className = 'source-main'; const name = document.createElement('span'); name.className = 'source-name'; name.textContent = source.source_id; const meta = document.createElement('span'); meta.className = 'source-meta'; meta.textContent = `${source.chunks} 个片段 · ${source.enabled ? '参与检索' : '已停用'}`; main.append(name, meta);
        const toggle = document.createElement('button'); toggle.className = 'secondary small'; toggle.textContent = source.enabled ? '停用' : '启用'; toggle.addEventListener('click', async () => { toggle.disabled = true; const response = await fetch(`/sources/${encodeURIComponent(source.source_id)}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({enabled:!source.enabled})}); if (!response.ok) alert(await readError(response)); await refreshStatus(); });
        const remove = document.createElement('button'); remove.className = 'danger small'; remove.textContent = '删除'; remove.addEventListener('click', async () => { if (!confirm(`确定删除“${source.source_id}”及其全部索引吗？`)) return; remove.disabled = true; const response = await fetch(`/sources/${encodeURIComponent(source.source_id)}`, {method:'DELETE'}); if (!response.ok) alert(await readError(response)); await refreshStatus(); }); row.append(icon, main, toggle, remove); list.append(row); });
    } catch { byId('serviceStatus').textContent = '服务未连接'; }
  }
  byId('files').addEventListener('change', event => { const count = event.target.files.length; byId('fileHint').textContent = count ? `已选择 ${count} 个文件` : '可同时上传多份资料'; });
  byId('uploadButton').addEventListener('click', async () => { const button = byId('uploadButton'); const message = byId('uploadMessage'); const selected = byId('files').files; message.className = 'message'; if (!selected.length) { message.textContent = '请先选择至少一个文件。'; return; } const body = new FormData(); Array.from(selected).forEach(file => body.append('files', file)); button.disabled = true; message.textContent = '正在解析并建立索引…'; try { const response = await fetch('/ingest/files', {method:'POST', body}); if (!response.ok) throw new Error(await readError(response)); const results = await response.json(); const chunks = results.reduce((total, item) => total + item.chunks_added, 0); message.textContent = `上传成功：${results.length} 份资料，新增 ${chunks} 个文本片段。`; await refreshStatus(); } catch (error) { message.className = 'message error'; message.textContent = `上传失败：${error.message}`; } finally { button.disabled = false; } });
  byId('reindexButton').addEventListener('click', async () => { const button = byId('reindexButton'); button.disabled = true; button.textContent = '正在重建…'; try { const response = await fetch('/reindex', {method:'POST'}); if (!response.ok) throw new Error(await readError(response)); const value = await response.json(); button.textContent = `已重建 ${value.active_chunks} 个片段`; await refreshStatus(); } catch (error) { alert(`重建失败：${error.message}`); } finally { button.disabled = false; setTimeout(() => { button.textContent = '重新建立索引'; }, 1800); } });
  function asPercent(value) { return `${Math.round((value || 0) * 100)}%`; }
  function renderEvaluation(report) { if (!report || report.status !== 'complete') return; byId('metricPass').textContent = `${report.passed_cases}/${report.cases}`; byId('metricGeneration').textContent = asPercent(report.generation_success_rate); byId('metricCoverage').textContent = asPercent(report.source_coverage); byId('metricRecall').textContent = asPercent(report.key_point_recall); byId('metricCitation').textContent = asPercent(report.citation_validity); byId('evaluationMessage').textContent = `最近评测：平均用时 ${(report.mean_latency_ms / 1000).toFixed(2)} 秒 · ${report.reranker_enabled ? '已启用 BGE 专用重排' : '未启用专用重排'}`; const details = byId('evaluationDetails'); details.replaceChildren(); report.details.forEach((item, index) => { const row = document.createElement('div'); row.className = 'evaluation-detail'; row.textContent = `${item.passed ? '✓' : '△'} 用例 ${index + 1}：要点 ${asPercent(item.key_point_recall)}，来源 ${asPercent(item.source_coverage)}，引用${item.citation_valid ? '有效' : '需检查'}。`; details.append(row); }); }
  async function refreshEvaluation() { try { const response = await fetch('/evaluation/showcase'); renderEvaluation(await response.json()); } catch { byId('evaluationMessage').textContent = '尚未读取到评测报告。'; } }
  byId('evaluationButton').addEventListener('click', async () => { if (!confirm('评测只使用项目自带的原创样例资料，预计调用 DeepSeek 约 8 次。确定运行吗？')) return; const button = byId('evaluationButton'); button.disabled = true; byId('evaluationMessage').textContent = '正在运行 4 个公开样例用例，可能需要约 1 分钟…'; try { const response = await fetch('/evaluation/showcase/run', {method:'POST'}); if (!response.ok) throw new Error(await readError(response)); renderEvaluation(await response.json()); } catch (error) { byId('evaluationMessage').textContent = `评测失败：${error.message}`; } finally { button.disabled = false; } });
  async function askQuestion() {
    const button = byId('askButton'); const question = byId('question').value.trim(); if (!question || button.disabled) return; byId('welcome').style.display = 'none'; addUserMessage(question); byId('question').value = ''; byId('question').style.height = 'auto'; const loading = addLoadingMessage(); button.disabled = true; byId('askMessage').textContent = '正在执行检索、重排与引用校验…';
    try { const response = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question, top_k:6, answer_mode:byId('answerMode').value, history:conversation.slice(-4)})}); if (!response.ok) throw new Error(await readError(response)); const value = await response.json(); renderAssistantMessage(loading, value); byId('askMessage').textContent = value.refused ? '资料中的证据不足，系统已拒绝无依据作答。' : '回答已完成，可展开查看检索过程和引用证据。'; if (!value.refused) { conversation.push({question, answer:value.answer}); if (conversation.length > 6) conversation.shift(); } }
    catch (error) { clearInterval(stageTimer); loading.remove(); const failure = document.createElement('article'); failure.className = 'message-row assistant'; failure.innerHTML = '<div class="avatar">AI</div><div><div class="message-head"><strong>请求失败</strong></div><div class="answer-warning"></div></div>'; failure.querySelector('.answer-warning').textContent = error.message; byId('chatMessages').append(failure); byId('askMessage').textContent = '本次提问失败，请检查服务状态后重试。'; scrollChat(); }
    finally { button.disabled = false; byId('question').focus(); }
  }
  document.querySelectorAll('.starter').forEach(button => button.addEventListener('click', () => { byId('answerMode').value = button.dataset.mode; byId('question').value = button.dataset.question; askQuestion(); }));
  byId('askButton').addEventListener('click', askQuestion);
  byId('question').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); askQuestion(); } });
  byId('question').addEventListener('input', event => { event.target.style.height = 'auto'; event.target.style.height = `${Math.min(event.target.scrollHeight,150)}px`; });
  byId('clearButton').addEventListener('click', () => { conversation.length = 0; byId('chatMessages').replaceChildren(); byId('welcome').style.display = 'block'; byId('askMessage').textContent = '回答仅依据已启用资料；智能模式会将相关片段发送给 DeepSeek。'; byId('question').focus(); });
  refreshStatus(); refreshEvaluation();
</script>
</body>
</html>
"""
