/* ═══════════════════════════════════════════════════════════════════
   交通流 AI 分析决策系统 — 前端交互引擎
   SSE 实时流（POST + ReadableStream 解析）· 工具步骤卡 · 思考流光
   ═══════════════════════════════════════════════════════════════════ */
'use strict';

/* ── Markdown + 代码高亮渲染 ─────────────────────────────── */
function esc(s){return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function escAttr(s){return esc(s).replace(/"/g,'&quot;').replace(/'/g,'&#39;')}
function slugLang(lang){
  return String(lang||'text').trim().toLowerCase().replace(/[^a-z0-9_+#.-]/g,'')||'text';
}
function highlightCode(code,lang){
  const raw=esc(code);
  const l=slugLang(lang);
  if(['json','jsonc'].includes(l)){
    return raw
      .replace(/("(?:\\.|[^"\\])*")(\s*:)?/g,(_,s,colon)=>colon?`<span class="tok-key">${s}</span>${colon}`:`<span class="tok-str">${s}</span>`)
      .replace(/\b(true|false|null)\b/g,'<span class="tok-lit">$1</span>')
      .replace(/(^|[^\w.-])(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)/gi,'$1<span class="tok-num">$2</span>');
  }
  if(['html','xml','svg'].includes(l)){
    return raw
      .replace(/(&lt;\/?)([\w:-]+)/g,'$1<span class="tok-key">$2</span>')
      .replace(/\s([\w:-]+)=/g,' <span class="tok-attr">$1</span>=')
      .replace(/(".*?")/g,'<span class="tok-str">$1</span>');
  }
  if(['css','scss','sass'].includes(l)){
    return raw
      .replace(/(\/\*[\s\S]*?\*\/)/g,'<span class="tok-comment">$1</span>')
      .replace(/([.#]?[-_a-zA-Z][-_a-zA-Z0-9]*)(\s*\{)/g,'<span class="tok-key">$1</span>$2')
      .replace(/([-\w]+)(\s*:)/g,'<span class="tok-attr">$1</span>$2')
      .replace(/(#(?:[0-9a-fA-F]{3}){1,2}\b|-?\d+(?:\.\d+)?(?:px|rem|em|%|vh|vw)?\b)/g,'<span class="tok-num">$1</span>');
  }
  const keywordSets={
    py:'and|as|assert|async|await|break|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield',
    python:'and|as|assert|async|await|break|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield',
    js:'async|await|break|case|catch|class|const|continue|default|else|export|extends|finally|for|from|function|if|import|let|new|null|return|switch|throw|true|try|typeof|undefined|var|while|yield',
    javascript:'async|await|break|case|catch|class|const|continue|default|else|export|extends|finally|for|from|function|if|import|let|new|null|return|switch|throw|true|try|typeof|undefined|var|while|yield',
    ts:'async|await|break|case|catch|class|const|continue|default|else|export|extends|finally|for|from|function|if|import|interface|let|new|null|return|switch|throw|true|try|type|typeof|undefined|var|while|yield',
    typescript:'async|await|break|case|catch|class|const|continue|default|else|export|extends|finally|for|from|function|if|import|interface|let|new|null|return|switch|throw|true|try|type|typeof|undefined|var|while|yield',
    sh:'case|do|done|elif|else|esac|fi|for|function|if|in|then|while',
    bash:'case|do|done|elif|else|esac|fi|for|function|if|in|then|while',
  };
  const stash=[];
  const save=html=>{
    const token=`@@TOK_${stash.length}@@`;
    stash.push(html);
    return token;
  };
  let html=raw
    .replace(/(#.*$|\/\/.*$)/gm,m=>save(`<span class="tok-comment">${m}</span>`))
    .replace(/("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g,m=>save(`<span class="tok-str">${m}</span>`))
    .replace(/(^|[^\w.-])(-?\d+(?:\.\d+)?)/g,(_,prefix,num)=>prefix+save(`<span class="tok-num">${num}</span>`));
  const words=keywordSets[l];
  if(words)html=html.replace(new RegExp(`\\b(${words})\\b`,'g'),'<span class="tok-key">$1</span>');
  return html.replace(/@@TOK_(\d+)@@/g,(_,i)=>stash[Number(i)]||'');
}
function codeBlockHtml(code,lang){
  const l=slugLang(lang);
  const label=l==='text'?'代码':l;
  return `<figure class="code-block" data-lang="${escAttr(l)}">`
    +`<figcaption><span>${esc(label)}</span></figcaption>`
    +`<pre><code class="language-${escAttr(l)}">${highlightCode(code,l)}</code></pre>`
    +`</figure>`;
}
function protectCodeBlocks(src){
  const stash=[];
  const save=(lang,code)=>{
    const token=`@@CODEBLOCK_${stash.length}@@`;
    stash.push(codeBlockHtml(code.replace(/\n$/,''),lang));
    return token;
  };
  const text=String(src||'')
    .replace(/\r\n?/g,'\n')
    .replace(/^```([^\n`]*)\n([\s\S]*?)^```\s*$/gm,(_,lang,code)=>save(lang,code))
    .replace(/^~~~([^\n~]*)\n([\s\S]*?)^~~~\s*$/gm,(_,lang,code)=>save(lang,code));
  return {text,stash};
}
function restoreCodeBlocks(html,stash){
  return html.replace(/@@CODEBLOCK_(\d+)@@/g,(_,i)=>stash[Number(i)]||'');
}
function katexHtml(tex,display){
  const raw=String(tex||'').trim();
  if(!raw)return '';
  if(window.katex&&window.katex.renderToString){
    try{
      return window.katex.renderToString(raw,{
        displayMode:!!display,
        throwOnError:false,
        strict:false,
        trust:false,
      });
    }catch(e){/* fallback below */}
  }
  return display
    ? `<div class="math-fallback">${esc(raw)}</div>`
    : `<span class="math-fallback">${esc(raw)}</span>`;
}
function protectMath(src){
  const stash=[];
  const save=(tex,display)=>{
    const token=`@@MATH_${stash.length}@@`;
    stash.push(katexHtml(tex,display));
    return token;
  };
  let text=src
    .replace(/\$\$([\s\S]+?)\$\$/g,(_,tex)=>save(tex,true))
    .replace(/\\\[([\s\S]+?)\\\]/g,(_,tex)=>save(tex,true))
    .replace(/\\\(([\s\S]+?)\\\)/g,(_,tex)=>save(tex,false));
  text=text.replace(/(^|[^\\$])\$([^\n$]*[\\_^=+\-*/][^\n$]*)\$/g,
    (_,prefix,tex)=>prefix+save(tex,false));
  return {text,stash};
}
function restoreMath(html,stash){
  return html.replace(/@@MATH_(\d+)@@/g,(_,i)=>stash[Number(i)]||'');
}
function inline(s){
  const stash=[];
  let text=String(s??'').replace(/`([^`\n]+)`/g,(_,code)=>{
    const token=`@@INLINECODE_${stash.length}@@`;
    stash.push(`<code>${esc(code)}</code>`);
    return token;
  });
  text=esc(text)
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g,'$1<em>$2</em>');
  return text.replace(/@@INLINECODE_(\d+)@@/g,(_,i)=>stash[Number(i)]||'');
}
function md(src){
  const protectedCode=protectCodeBlocks(src);
  const protectedMath=protectMath(protectedCode.text);
  const lines=protectedMath.text.split('\n');let out='',i=0;
  while(i<lines.length){
    const L=lines[i];
    if(/^\s*$/.test(L)){i++;continue}
    if(/^@@CODEBLOCK_\d+@@$/.test(L.trim())){out+=L.trim();i++;continue}
    if(/^### /.test(L)){out+='<h3>'+inline(L.slice(4))+'</h3>';i++;continue}
    if(/^## /.test(L)){out+='<h2>'+inline(L.slice(3))+'</h2>';i++;continue}
    if(/^# /.test(L)){out+='<h1>'+inline(L.slice(2))+'</h1>';i++;continue}
    if(/^> /.test(L)){out+='<blockquote>'+inline(L.slice(2))+'</blockquote>';i++;continue}
    if(/^---+\s*$/.test(L)){out+='<hr>';i++;continue}
    if(/^\|/.test(L)){
      const rows=[];while(i<lines.length&&/^\|/.test(lines[i])){rows.push(lines[i]);i++}
      const cells=r=>r.replace(/^\||\|$/g,'').split('|').map(c=>inline(c.trim()));
      let t='<table>';
      rows.forEach((r,ri)=>{
        if(/---/.test(r)&&/^[\s|:-]+$/.test(r))return;
        const tag=ri===0?'th':'td';
        t+='<tr>'+cells(r).map(c=>`<${tag}>${c}</${tag}>`).join('')+'</tr>';
      });
      out+=t+'</table>';continue}
    if(/^[-•] /.test(L)){
      let l='<ul>';while(i<lines.length&&/^[-•] /.test(lines[i])){l+='<li>'+inline(lines[i].slice(2))+'</li>';i++}
      out+=l+'</ul>';continue}
    if(/^\d+\. /.test(L)){
      let l='<ol>';while(i<lines.length&&/^\d+\. /.test(lines[i])){l+='<li>'+inline(lines[i].replace(/^\d+\. /,''))+'</li>';i++}
      out+=l+'</ol>';continue}
    const para=[L];i++;
    while(i<lines.length&&!/^\s*$/.test(lines[i])&&!/^(#{1,3} |> |---+\s*$|\||[-•] |\d+\. |@@CODEBLOCK_\d+@@$)/.test(lines[i])){
      para.push(lines[i]);i++;
    }
    out+='<p>'+inline(para.join('\n'))+'</p>';
  }
  return restoreCodeBlocks(restoreMath(out,protectedMath.stash),protectedCode.stash);
}

/* ── 基础工具 ───────────────────────────────────────────── */
const $=id=>document.getElementById(id);
const thread=$('thread'),scrollBox=$('scroll'),input=$('input');
let busy=false;
let CONTEXT_STATUS=null;

function scrollBottom(){scrollBox.scrollTop=scrollBox.scrollHeight}
function now(){return new Date().toTimeString().slice(0,8)}
setInterval(()=>{$('clock').textContent=now()},1000);$('clock').textContent=now();
const sleep=ms=>new Promise(r=>setTimeout(r,ms));

function setBusy(b){
  busy=b;
  $('btnSend').disabled=b;$('btnReport').disabled=b;$('btnReset').disabled=b;
  document.querySelectorAll('.qchip').forEach(c=>c.disabled=b);
  ['modelSelect','thinkingSelect'].forEach(id=>{const el=$(id);if(el)el.disabled=b});
  document.querySelectorAll('.setting-pill').forEach(btn=>btn.disabled=b);
  if(b)closeSettingMenus();
}

/* ── SSE over fetch ─────────────────────────────────────── */
async function ssePost(url,body,onEvent){
  const resp=await fetch(url,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body||{}),
  });
  const ct=resp.headers.get('Content-Type')||'';
  if(!resp.ok||!ct.includes('text/event-stream')){
    let msg=`请求失败（HTTP ${resp.status}）`;
    try{const j=await resp.json();if(j.error)msg=j.error}catch(e){/* ignore */}
    throw new Error(msg);
  }
  const reader=resp.body.getReader();
  const dec=new TextDecoder();
  let buf='';
  for(;;){
    const {done,value}=await reader.read();
    if(done)break;
    buf+=dec.decode(value,{stream:true});
    let idx;
    while((idx=buf.indexOf('\n\n'))>=0){
      const chunk=buf.slice(0,idx);buf=buf.slice(idx+2);
      const line=chunk.split('\n').find(l=>l.startsWith('data:'));
      if(!line)continue;
      try{onEvent(JSON.parse(line.slice(5).trim()))}catch(e){/* 忽略坏帧 */}
    }
  }
}

/* ── 消息 DOM 构建 ──────────────────────────────────────── */
function hideHero(){const h=$('hero');if(h)h.style.display='none'}

function addUser(text){
  const div=document.createElement('div');div.className='msg user';
  div.innerHTML=`<div class="avatar">我</div><div class="bubble">${esc(text)}<div class="meta" style="text-align:right">${now()}</div></div>`;
  thread.appendChild(div);scrollBottom();
}
function addAgentShell(){
  const div=document.createElement('div');div.className='msg agent';
  div.innerHTML=`<div class="avatar">AI</div><div class="bubble"><div class="stream"></div><span class="cursor"></span><div class="meta">${MODEL} · ${now()}</div></div>`;
  thread.appendChild(div);scrollBottom();
  return div;
}
function addError(message){
  const div=document.createElement('div');div.className='msg errormsg';
  div.innerHTML=`<div class="avatar">!</div><div class="bubble">${esc(message)}<div class="meta">${now()}</div></div>`;
  thread.appendChild(div);scrollBottom();
}
function addStepsBox(){
  const box=document.createElement('div');box.className='steps';
  thread.appendChild(box);scrollBottom();return box;
}
function highlightJson(s){
  return esc(s)
    .replace(/"([^"]+)":/g,'<span class="k">"$1"</span>:')
    .replace(/([:\s\[{,])(-?\d+\.?\d*)/g,'$1<span class="n">$2</span>');
}
function addStep(box,label){
  const el=document.createElement('div');el.className='step running';
  el.innerHTML=`<div class="step-head"><span class="sicon"></span>
    <span class="step-label">${esc(label)}</span>
    <span class="step-time">RUNNING</span></div>
    <div class="step-body"></div><div class="step-scan"></div>`;
  el.addEventListener('click',()=>el.classList.toggle('open'));
  box.appendChild(el);scrollBottom();return el;
}
function formatDuration(seconds){
  const s=Number(seconds);
  if(!Number.isFinite(s)||s<0)return '—';
  if(s<0.1)return Math.max(1,Math.round(s*1000))+'ms';
  if(s<10)return s.toFixed(1)+'s';
  return Math.round(s)+'s';
}
function finishStep(el,seconds,resultPreview){
  el.classList.remove('running');el.classList.add('done');
  el.querySelector('.step-time').textContent=formatDuration(seconds);
  el.querySelector('.step-body').innerHTML=highlightJson(resultPreview||'');
}
let thinkingEl=null;
function showThinking(text){
  clearThinking();
  thinkingEl=document.createElement('div');thinkingEl.className='thinking';
  thinkingEl.innerHTML=`<span class="think-text">${esc(text)}</span>`;
  thread.appendChild(thinkingEl);scrollBottom();
}
function clearThinking(){if(thinkingEl){thinkingEl.remove();thinkingEl=null}}

/* ── 对话流状态机 ───────────────────────────────────────── */
function makeChatRenderer(){
  let bubble=null,streamText='',stepsBox=null;
  const stepMap=new Map();  // call_id -> {el, t0}

  function finalizeBubble(){
    if(!bubble)return;
    const cur=bubble.querySelector('.cursor');if(cur)cur.remove();
    bubble=null;streamText='';
  }
  return {
    onEvent(ev){
      if(ev.type==='text'){
        clearThinking();
        if(!bubble){bubble=addAgentShell();streamText='';stepsBox=null}
        streamText+=ev.text;
        bubble.querySelector('.stream').innerHTML=md(streamText);
        scrollBottom();
      }else if(ev.type==='tool_call'){
        clearThinking();finalizeBubble();
        if(!stepsBox)stepsBox=addStepsBox();
        const el=addStep(stepsBox,ev.label||ev.name);
        stepMap.set(ev.call_id,{el,t0:performance.now()});
      }else if(ev.type==='tool_result'){
        const rec=stepMap.get(ev.call_id);
        if(rec){
          const elapsed=typeof ev.elapsed_s==='number' ? ev.elapsed_s : (performance.now()-rec.t0)/1000;
          finishStep(rec.el,elapsed,ev.preview);
        }
        showThinking('正在整合数据 · 组织分析结论 …');
      }else if(ev.type==='context'){
        updateContextMeter(ev.context);
      }else if(ev.type==='error'){
        clearThinking();finalizeBubble();addError(ev.message);
      }else if(ev.type==='done'){
        clearThinking();finalizeBubble();
      }
    },
    finish(){clearThinking();finalizeBubble()},
  };
}

async function send(text){
  const t=(text||input.value).trim();
  if(!t||busy)return;
  hideCommandPalette();
  input.value='';input.style.height='62px';
  if(t.startsWith('/')){
    hideHero();addUser(t);
    await dispatchSlashCommand(t);
    return;
  }
  setBusy(true);hideHero();
  addUser(t);
  showThinking('正在理解问题 · 规划分析步骤 …');
  const renderer=makeChatRenderer();
  try{
    await ssePost('/api/agent/chat',{message:t},renderer.onEvent);
  }catch(e){
    clearThinking();addError(e.message||'连接失败，请检查服务是否运行');
  }finally{
    renderer.finish();setBusy(false);
  }
}

$('btnSend').addEventListener('click',()=>send());
input.addEventListener('keydown',e=>{
  if(commandPaletteOpen()){
    if(e.key==='ArrowDown'){e.preventDefault();moveCommandSelection(1);return}
    if(e.key==='ArrowUp'){e.preventDefault();moveCommandSelection(-1);return}
    if(e.key==='Escape'){e.preventDefault();hideCommandPalette();return}
    if(e.key==='Tab'){e.preventDefault();acceptCommandSelection();return}
    if(e.key==='Enter'&&!e.shiftKey){
      e.preventDefault();
      if(!commandPaletteItems.length){hideCommandPalette();send()}
      else if(commandSelectionIsExact()){hideCommandPalette();send()}
      else acceptCommandSelection();
      return;
    }
  }
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}
});
input.addEventListener('input',()=>{
  input.style.height='62px';input.style.height=Math.min(150,input.scrollHeight)+'px';
  updateCommandPalette();
});
input.addEventListener('focus',updateCommandPalette);
input.addEventListener('blur',()=>setTimeout(hideCommandPalette,120));

/* ── 新会话 ─────────────────────────────────────────────── */
async function runClear(){
  try{await fetch('/api/agent/reset',{method:'POST'})}catch(e){/* ignore */}
  thread.querySelectorAll('.msg,.steps,.thinking').forEach(el=>el.remove());
  const h=$('hero');if(h)h.style.display='';
}
$('btnReset').addEventListener('click',()=>{if(!busy)runClear()});

/* ── 报告抽屉 ───────────────────────────────────────────── */
const drawer=$('drawer'),overlay=$('overlay');
let reportGenerated=false;

function openDrawer(){overlay.classList.add('show');drawer.classList.add('show')}
function closeDrawer(){overlay.classList.remove('show');drawer.classList.remove('show')}
$('btnClose').addEventListener('click',closeDrawer);
overlay.addEventListener('click',closeDrawer);
$('btnDownload').addEventListener('click',()=>{window.location.href='/api/agent/report/download'});

async function runReportGeneration(){
  openDrawer();
  if(reportGenerated)return;   // 已有报告则只查看；新会话后可重新生成
  setBusy(true);
  const status=$('drawerStatus');
  status.textContent='正在生成报告 …';status.classList.add('shimmer');status.classList.remove('done');
  const body=$('drawerBody');body.innerHTML='';
  const holder=document.createElement('div');body.appendChild(holder);
  let text='';
  try{
    await ssePost('/api/agent/report',{},ev=>{
      if(ev.type==='text'){
        text+=ev.text;
        holder.innerHTML=md(text);
        body.scrollTop=body.scrollHeight;
      }else if(ev.type==='report_saved'){
        status.classList.remove('shimmer');
        status.textContent='已保存 → '+ev.path;
        status.classList.add('done');
        $('btnDownload').disabled=false;
        reportGenerated=true;
      }else if(ev.type==='error'){
        status.classList.remove('shimmer');
        status.textContent='生成失败';
        holder.innerHTML+=`<blockquote>${esc(ev.message)}</blockquote>`;
      }
    });
  }catch(e){
    status.classList.remove('shimmer');
    status.textContent='生成失败';
    holder.innerHTML+=`<blockquote>${esc(e.message||'连接失败')}</blockquote>`;
  }finally{
    setBusy(false);
  }
}
$('btnReport').addEventListener('click',()=>{
  if(busy){openDrawer();return}
  runReportGeneration();
});

/* ── 内部命令（/help /clear /report /summary /reload /tools /model）──── */
function addSystem(html){
  const div=document.createElement('div');div.className='msg systemmsg';
  div.innerHTML=`<div class="avatar">/</div><div class="bubble">${html}<div class="meta">${now()}</div></div>`;
  thread.appendChild(div);scrollBottom();
  return div;
}

const SLASH_COMMANDS={
  help:   {desc:'显示全部内部命令'},
  clear:  {desc:'清空当前会话（不影响已生成的报告）'},
  report: {desc:'生成完整分析报告（等同点击左下角按钮）'},
  summary:{desc:'显示当前数据概览 KPI'},
  reload: {desc:'重新加载 outputs/ 检测数据'},
  tools:  {desc:'列出智能体当前可调用的全部分析工具'},
  model:  {desc:'显示当前使用的模型与连接状态'},
};

let commandPaletteEl=null;
let commandPaletteItems=[];
let commandPaletteIndex=0;
let commandPaletteLastQuery=null;

function commandQuery(){
  const raw=input.value;
  if(!raw.startsWith('/'))return null;
  if(/\s/.test(raw))return null;
  return raw.slice(1).toLowerCase();
}
function ensureCommandPalette(){
  if(commandPaletteEl)return commandPaletteEl;
  const box=document.createElement('div');
  box.className='command-palette';
  box.setAttribute('role','listbox');
  box.setAttribute('aria-label','可用命令');
  input.parentElement.appendChild(box);
  commandPaletteEl=box;
  return box;
}
function commandPaletteOpen(){
  return !!(commandPaletteEl&&commandPaletteEl.classList.contains('show'));
}
function hideCommandPalette(){
  if(commandPaletteEl)commandPaletteEl.classList.remove('show');
}
function commandMatches(query){
  return Object.entries(SLASH_COMMANDS)
    .filter(([name,c])=>!query||name.includes(query)||c.desc.includes(query))
    .map(([name,c])=>({name,desc:c.desc}));
}
function renderCommandPalette(){
  const box=ensureCommandPalette();
  if(!commandPaletteItems.length){
    box.innerHTML='<div class="cmd-empty">没有匹配的命令</div>';
    box.classList.add('show');
    return;
  }
  box.innerHTML=commandPaletteItems.map((item,idx)=>`
    <button type="button" class="cmd-item${idx===commandPaletteIndex?' active':''}"
      data-command="${escAttr(item.name)}" role="option" aria-selected="${idx===commandPaletteIndex?'true':'false'}">
      <span class="cmd-name">/${esc(item.name)}</span>
      <span class="cmd-desc">${esc(item.desc)}</span>
    </button>
  `).join('');
  box.querySelectorAll('.cmd-item').forEach((btn,idx)=>{
    btn.addEventListener('mousedown',e=>e.preventDefault());
    btn.addEventListener('click',()=>{
      commandPaletteIndex=idx;
      acceptCommandSelection();
    });
  });
  box.classList.add('show');
}
function updateCommandPalette(){
  const query=commandQuery();
  if(query===null||busy){hideCommandPalette();return}
  if(query!==commandPaletteLastQuery){
    commandPaletteIndex=0;
    commandPaletteLastQuery=query;
  }
  commandPaletteItems=commandMatches(query);
  commandPaletteIndex=Math.min(commandPaletteIndex,Math.max(0,commandPaletteItems.length-1));
  renderCommandPalette();
}
function moveCommandSelection(delta){
  if(!commandPaletteItems.length)return;
  commandPaletteIndex=(commandPaletteIndex+delta+commandPaletteItems.length)%commandPaletteItems.length;
  renderCommandPalette();
}
function acceptCommandSelection(){
  if(!commandPaletteItems.length)return false;
  const item=commandPaletteItems[commandPaletteIndex];
  input.value='/'+item.name;
  input.focus();
  input.setSelectionRange(input.value.length,input.value.length);
  hideCommandPalette();
  return true;
}
function commandSelectionIsExact(){
  if(!commandPaletteItems.length)return false;
  return input.value.trim().toLowerCase()==='/'+commandPaletteItems[commandPaletteIndex].name;
}

function cmdHelp(){
  const lines=Object.entries(SLASH_COMMANDS)
    .map(([name,c])=>`<li><code>/${name}</code> — ${esc(c.desc)}</li>`).join('');
  addSystem(`<strong>内部命令</strong><ul>${lines}</ul>`);
}

async function cmdSummary(){
  let s;
  try{s=await fetchJson('/api/agent/summary')}catch(e){addSystem('获取数据概览失败。');return}
  if(!s||!s.data_ready){addSystem('暂无检测数据，请先运行检测生成 outputs/ 数据。');return}
  addSystem(`<strong>数据概览</strong><table>
    <tr><td>轨迹总数</td><td>${s.unique_tracks??'—'}</td></tr>
    <tr><td>过线事件</td><td>${s.total_events??'—'}</td></tr>
    <tr><td>折算小时流量</td><td>${s.hourly_flow??'—'} 辆/h</td></tr>
    <tr><td>平均速度</td><td>${s.avg_speed_kmh??'—'} km/h</td></tr>
    <tr><td>85 分位速度</td><td>${s.p85_speed_kmh??'—'} km/h</td></tr>
    <tr><td>高峰分钟流量</td><td>${(s.peak_minute&&s.peak_minute.count)??'—'} 辆/min</td></tr>
  </table>`);
}

async function cmdReload(){
  let resp,r;
  try{resp=await fetch('/api/agent/reload',{method:'POST'});r=await resp.json()}
  catch(e){addSystem('重新加载失败，请检查服务是否运行。');return}
  if(r&&r.ok){renderSummary(r.summary);addSystem('已重新加载 outputs/ 检测数据，侧栏 KPI 已刷新。')}
  else addSystem('重新加载失败。');
}

async function cmdTools(){
  let list;
  try{list=await fetchJson('/api/agent/tools')}catch(e){addSystem('获取工具列表失败。');return}
  const lines=list.map(t=>
    `<li><strong>${esc(t.label)}</strong> <span style="color:var(--text-dim)">${esc(t.name)}</span></li>`
  ).join('');
  addSystem(`<strong>已接入的分析工具（共 ${list.length} 个）</strong><ul>${lines}</ul>`);
}

async function cmdModel(){
  let h;
  try{h=await fetchJson('/api/agent/health')}catch(e){addSystem('获取状态失败。');return}
  addSystem(`模型：<code>${esc(h.model)}</code><br>`
    +`思考强度：<code>${esc(h.thinking_strength||'medium')}</code><br>`
    +`Key 已配置：${h.key_configured?'是':'否'}　数据就绪：${h.data_ready?'是':'否'}`);
}

async function dispatchSlashCommand(raw){
  const [name,...rest]=raw.slice(1).trim().split(/\s+/);
  const key=(name||'').toLowerCase();
  if(!(key in SLASH_COMMANDS)){
    addSystem(`未知命令 <code>/${esc(key)}</code>。输入 <code>/help</code> 查看全部命令。`);
    return;
  }
  if(key==='report'){await runReportGeneration();return}
  setBusy(true);
  try{
    if(key==='clear'){await runClear();addSystem('会话已清空。')}
    else if(key==='help')cmdHelp();
    else if(key==='summary')await cmdSummary();
    else if(key==='reload')await cmdReload();
    else if(key==='tools')await cmdTools();
    else if(key==='model')await cmdModel();
  }catch(e){
    addSystem(`命令执行失败：${esc(e.message||String(e))}`);
  }finally{
    setBusy(false);
  }
}

/* ── 侧栏 KPI 渲染 ──────────────────────────────────────── */
const CLASS_COLORS={'小汽车':'#34d399','摩托/电动车':'#c084fc','货车':'#60a5fa',
                    '公交/大客车':'#f59e0b','自行车':'#22d3ee'};
const ENTRANCE_COLORS={'南进口':'#22d3ee','北进口':'#6366f1','东进口':'#60a5fa'};

function kpiCell(value,name,unit,alt,dec){
  return `<div class="kpi${alt?' alt':''}"><div class="kpi-num" data-count="${value}" data-dec="${dec||0}">0${unit?`<small>${unit}</small>`:''}</div><div class="kpi-name">${name}</div></div>`;
}
function barRows(counts,colors){
  const entries=Object.entries(counts);
  if(!entries.length)return '<div class="quality-note">暂无数据</div>';
  const max=Math.max(...entries.map(([,v])=>v));
  return entries.map(([k,v])=>
    `<div class="bar-row"><span>${esc(k)}</span><div class="bar-track"><div class="bar-fill" data-w="${Math.max(1,Math.round(v/max*100))}" style="background:${colors[k]||'#8aa4ba'}"></div></div><span class="bar-val">${v}</span></div>`
  ).join('');
}
function countUp(el){
  const target=parseFloat(el.dataset.count)||0,dec=parseInt(el.dataset.dec||'0');
  const small=el.querySelector('small');const suffix=small?small.outerHTML:'';
  const t0=performance.now(),dur=1100;
  function tick(t){
    const p=Math.min(1,(t-t0)/dur),e=1-Math.pow(1-p,3);
    el.innerHTML=(target*e).toFixed(dec).replace(/\B(?=(\d{3})+(?!\d))/g,',')+suffix;
    if(p<1)requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
function renderSummary(s){
  if(!s||!s.data_ready){
    $('kpiGrid').innerHTML='<div class="quality-note">暂无检测数据</div>';
    $('classBars').innerHTML='<div class="quality-note">请先运行检测生成 outputs/ 数据</div>';
    $('entranceBars').innerHTML='';
    $('qcNote').textContent='—';
    return;
  }
  $('kpiGrid').innerHTML=[
    kpiCell(s.unique_tracks||0,'轨迹总数','',false,0),
    kpiCell(s.total_events||0,'过线事件','',false,0),
    kpiCell(s.hourly_flow||0,'折算小时流量','辆/h',true,0),
    kpiCell(s.avg_speed_kmh||0,'平均断面速度','km/h',true,1),
    kpiCell(s.p85_speed_kmh||0,'85 分位速度','km/h',true,1),
    kpiCell((s.peak_minute&&s.peak_minute.count)||0,'高峰分钟流量','辆/min',true,0),
  ].join('');
  $('classBars').innerHTML=barRows(s.class_counts||{},CLASS_COLORS);
  $('entranceBars').innerHTML=barRows(s.entrance_counts||{},ENTRANCE_COLORS);
  const qc=s.qc||{};
  $('qcNote').innerHTML=`已自动清洗：<em>${qc.speed_outliers||0}</em> 条超速离群（标定畸变）· `
    +`零速事件 <em>${qc.zero_speed||0}</em> 条单列 · 对向/未识别车道不计入车道统计`;
  document.querySelectorAll('[data-count]').forEach(countUp);
  setTimeout(()=>{document.querySelectorAll('.bar-fill').forEach(b=>{b.style.width=b.dataset.w+'%'})},200);
}

/* ── 初始化 ─────────────────────────────────────────────── */
let MODEL='deepseek-v4-flash';
let THINKING_STRENGTH='medium';

async function fetchJson(url){
  const r=await fetch(url);
  if(!r.ok)throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function thinkingLabel(value,options){
  const found=(options||[]).find(o=>o.value===value);
  return found ? found.label : value;
}
function fmtTokens(n){
  const v=Number(n)||0;
  if(v>=1000)return (v/1000).toFixed(v>=10000?0:1)+'k';
  return String(Math.round(v));
}
function updateContextMeter(ctx){
  if(!ctx)return;
  CONTEXT_STATUS=ctx;
  const meter=$('contextMeter');
  if(!meter)return;
  const ratio=Math.max(0,Math.min(1,Number(ctx.ratio)||0));
  meter.style.setProperty('--ctx',Math.round(ratio*360)+'deg');
  meter.classList.toggle('warn',ratio>=0.7&&ratio<0.9);
  meter.classList.toggle('danger',ratio>=0.9);
  const pct=Math.round(ratio*100);
  $('contextMeterText').textContent=pct+'%';
  $('contextUsed').textContent=fmtTokens(ctx.used_tokens)+' tokens';
  $('contextLimit').textContent=fmtTokens(ctx.limit_tokens)+' tokens';
  $('contextRatio').textContent=pct+'%';
  $('contextBar').style.width=pct+'%';
}
function showContextStatus(){
  if(!CONTEXT_STATUS){addSystem('上下文窗口：暂无会话统计。');return}
  const c=CONTEXT_STATUS;
  const pct=Math.round((Number(c.ratio)||0)*100);
  addSystem(`上下文窗口：<code>${fmtTokens(c.used_tokens)}</code> / <code>${fmtTokens(c.limit_tokens)}</code> tokens（${pct}%）<br>`
    +`当前会话消息数：<code>${esc(c.message_count??0)}</code>。`);
}
function closeSettingMenus(){
  document.querySelectorAll('.setting-menu.open').forEach(el=>el.classList.remove('open'));
}
function settingOptionHtml(kind,item,current,icon){
  return `<button type="button" class="setting-option${item.value===current?' active':''}"
    data-kind="${escAttr(kind)}" data-value="${escAttr(item.value)}" role="menuitemradio"
    aria-checked="${item.value===current?'true':'false'}">
    <span class="setting-option-main">
      ${icon?`<span class="setting-option-icon">${esc(icon)}</span>`:''}
      <span class="setting-option-label">${esc(item.label||item.value)}</span>
    </span>
  </button>`;
}
function renderSettingMenu(models,thinkingOptions,currentModel,currentThinking){
  const menu=$('settingsMenu');
  if(!menu)return;
  menu.innerHTML=[
    '<div class="setting-group-title">推理</div>',
    ...thinkingOptions.map(item=>settingOptionHtml('thinking',item,currentThinking,'')),
    '<div class="setting-divider"></div>',
    '<div class="setting-group-title">模型</div>',
    ...models.map(item=>settingOptionHtml('model',item,currentModel,'⌁')),
  ].join('');
  menu.querySelectorAll('.setting-option').forEach(btn=>{
    btn.addEventListener('mousedown',e=>e.preventDefault());
    btn.addEventListener('click',()=>{
      const kind=btn.dataset.kind||'';
      const value=btn.dataset.value||'';
      closeSettingMenus();
      if(kind==='model'&&value&&value!==$('modelSelect').value){
        $('modelSelect').value=value;
        updateModelSettings(true);
      }else if(kind==='thinking'&&value&&value!==$('thinkingSelect').value){
        $('thinkingSelect').value=value;
        updateModelSettings(true);
      }
    });
  });
}
function renderModelSettings(health){
  const modelSel=$('modelSelect'),thinkingSel=$('thinkingSelect');
  if(!modelSel||!thinkingSel)return;
  const models=health.available_models&&health.available_models.length
    ? health.available_models : [health.model||MODEL];
  modelSel.innerHTML=models.map(m=>`<option value="${escAttr(m)}">${esc(m)}</option>`).join('');
  modelSel.value=health.model||models[0];
  const options=health.thinking_options&&health.thinking_options.length
    ? health.thinking_options
    : [{value:'low',label:'低'},{value:'medium',label:'中'},{value:'high',label:'高'},{value:'ultra',label:'超高'}];
  thinkingSel.innerHTML=options
    .map(o=>`<option value="${escAttr(o.value)}">${esc(o.label||o.value)}</option>`)
    .join('');
  thinkingSel.value=health.thinking_strength||THINKING_STRENGTH;
  $('settingsPillValue').textContent=`${modelSel.value} ${thinkingLabel(thinkingSel.value,options)}`;
  renderSettingMenu(models.map(m=>({value:m,label:m})),options,modelSel.value,thinkingSel.value);
}
async function updateModelSettings(showToast=true){
  if(busy)return;
  const model=$('modelSelect')?.value||MODEL;
  const thinking_strength=$('thinkingSelect')?.value||THINKING_STRENGTH;
  setBusy(true);
  try{
    const resp=await fetch('/api/agent/settings',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model,thinking_strength}),
    });
    const data=await resp.json();
    if(!resp.ok||!data.ok)throw new Error(data.error||`HTTP ${resp.status}`);
    const h=data.settings||{};
    MODEL=h.model||model;
    THINKING_STRENGTH=h.thinking_strength||thinking_strength;
    $('modelName').textContent=MODEL;
    renderModelSettings(h);
    if(showToast&&data.changed){
      const label=thinkingLabel(THINKING_STRENGTH,h.thinking_options);
      addSystem(`已切换模型：<code>${esc(MODEL)}</code>，思考强度：<code>${esc(label)}</code>。会话上下文已重置。`);
    }
  }catch(e){
    addSystem(`模型设置失败：${esc(e.message||String(e))}`);
    try{
      const h=await fetchJson('/api/agent/health');
      MODEL=h.model||MODEL;
      THINKING_STRENGTH=h.thinking_strength||THINKING_STRENGTH;
      $('modelName').textContent=MODEL;
      renderModelSettings(h);
    }catch(_e){/* ignore */}
  }finally{
    setBusy(false);
  }
}

async function init(){
  let health=null,summary=null,questions=[];
  try{
    [health,summary,questions]=await Promise.all([
      fetchJson('/api/agent/health'),
      fetchJson('/api/agent/summary'),
      fetchJson('/api/agent/quick_questions'),
    ]);
  }catch(e){
    $('heroText').textContent='无法连接后端服务，请重启应用';
    return;
  }
  MODEL=health.model||MODEL;
  THINKING_STRENGTH=health.thinking_strength||THINKING_STRENGTH;
  $('modelName').textContent=MODEL;
  renderModelSettings(health);
  updateContextMeter(health.context);
  $('contextMeter').addEventListener('click',showContextStatus);
  document.querySelectorAll('.setting-pill').forEach(btn=>{
    btn.addEventListener('click',e=>{
      e.stopPropagation();
      if(btn.disabled)return;
      const wrap=btn.closest('.setting-menu');
      const wasOpen=wrap.classList.contains('open');
      closeSettingMenus();
      if(!wasOpen)wrap.classList.add('open');
    });
  });
  document.addEventListener('click',closeSettingMenus);
  document.addEventListener('keydown',e=>{if(e.key==='Escape')closeSettingMenus()});

  if(!health.key_configured){
    $('modelDot').classList.add('warn');
    const b=$('banner');
    b.innerHTML='未检测到 DeepSeek API Key —— 请在终端执行 <code>export DEEPSEEK_API_KEY=sk-xxx</code> 后重启应用；数据概览不受影响，但 AI 对话与报告不可用';
    b.hidden=false;document.body.classList.add('has-banner');
  }
  $('dataChip').textContent=health.data_ready?'数据就绪 · 三进口':'数据未就绪';

  renderSummary(summary);

  if(summary&&summary.data_ready){
    $('heroText').innerHTML=`三进口检测数据已接入：<code>${summary.total_events}</code> 次过线事件 · <code>${summary.unique_tracks||'—'}</code> 条轨迹。<br>直接询问拥堵成因、信号配时、车道利用或异常清洗。`;
  }else{
    $('heroText').textContent='未找到检测数据：请先运行三进口检测（python3 -m src.main run-all）生成 outputs/ 数据后重启';
  }

  $('quickChips').innerHTML=(questions||[]).map(q=>`<button class="qchip">${esc(q)}</button>`).join('');
  document.querySelectorAll('.qchip').forEach(c=>c.addEventListener('click',()=>send(c.textContent)));

  if(health.report_exists){$('btnDownload').disabled=false}
}

init();
