const $ = id => document.getElementById(id);
const ui = Object.fromEntries(['file','choose','replace','drop-zone','empty','image','video','filename','media-meta','run','model','load','model-status','model-dot','top-status','prompt','interval','threshold','output','result-status'].map(id => [id, $(id)]));
let ready = false, mediaType = '', objectUrl = '', busy = false, previousFrame = null, lastSample = -Infinity, session = 0, lastCaption = '';

async function post(path, data) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || 'Request failed');
  return body;
}
function controls() { ui.run.disabled = !ready || !mediaType || busy; ui.run.firstChild.textContent = busy ? 'Generating... ' : mediaType === 'video' ? 'Start captioning ' : 'Generate caption '; }
async function status() {
  try {
    const s = await (await fetch('/api/status')).json();
    ready = s.phase === 'ready';
    ui['model-status'].textContent = s.message;
    ui['model-dot'].className = 'status-dot ' + (ready ? 'on' : s.phase === 'error' ? 'error' : '');
    ui['top-status'].textContent = ready ? 'Model ready' : s.phase === 'downloading' || s.phase === 'loading' ? 'Preparing model' : 'Local workspace';
    ui.load.disabled = s.phase === 'downloading' || s.phase === 'loading';
    ui.load.firstChild.textContent = ui.load.disabled ? 'Preparing model... ' : 'Download & load ';
    controls();
  } catch (_) { ui['model-status'].textContent = 'Cannot connect to local server.'; }
}
ui.load.onclick = async () => { try { await post('/api/model', {model:ui.model.value}); await status(); } catch (e) { ui['model-status'].textContent = e.message; } };
ui.choose.onclick = ui.replace.onclick = () => ui.file.click();
function selectFile(file) {
  if (!file || !/^(image|video)\//.test(file.type)) { ui['result-status'].textContent = 'CHOOSE AN IMAGE OR VIDEO'; return; }
  session++; ui.video.pause(); if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file); mediaType = file.type.startsWith('video/') ? 'video' : 'image';
  ui.image.hidden = mediaType !== 'image'; ui.video.hidden = mediaType !== 'video'; ui.empty.hidden = true; ui.replace.hidden = false;
  (mediaType === 'image' ? ui.image : ui.video).src = objectUrl;
  ui.filename.textContent = file.name; ui['media-meta'].textContent = mediaType.toUpperCase();
  ui.output.innerHTML = '<div class="output-empty"><span>✳</span><div><strong>Ready for a closer look</strong><p>Generate a caption to see the result.</p></div></div>';
  ui['result-status'].textContent = 'READY'; previousFrame = null; lastSample = -Infinity; lastCaption = ''; controls();
}
ui.file.onchange = () => selectFile(ui.file.files[0]);
for (const event of ['dragenter','dragover']) ui['drop-zone'].addEventListener(event, e => { e.preventDefault(); ui['drop-zone'].classList.add('drag'); });
for (const event of ['dragleave','drop']) ui['drop-zone'].addEventListener(event, e => { e.preventDefault(); ui['drop-zone'].classList.remove('drag'); });
ui['drop-zone'].addEventListener('drop', e => selectFile(e.dataTransfer.files[0]));
function frame(element) {
  const canvas = document.createElement('canvas'); const width = element.videoWidth || element.naturalWidth, height = element.videoHeight || element.naturalHeight;
  if (!width || !height) throw new Error('Media is not ready yet.');
  const scale = Math.min(1, 1024 / Math.max(width, height)); canvas.width = Math.max(1, Math.round(width * scale)); canvas.height = Math.max(1, Math.round(height * scale));
  const ctx = canvas.getContext('2d', {willReadFrequently:true}); ctx.drawImage(element, 0, 0, canvas.width, canvas.height);
  const small = document.createElement('canvas'); small.width = 32; small.height = 32; const sc = small.getContext('2d', {willReadFrequently:true}); sc.drawImage(canvas, 0, 0, 32, 32);
  return {image:canvas.toDataURL('image/jpeg', .85), pixels:sc.getImageData(0,0,32,32).data};
}
function difference(a,b) { let total = 0; for (let i=0;i<a.length;i+=4) total += Math.abs(a[i]-b[i])+Math.abs(a[i+1]-b[i+1])+Math.abs(a[i+2]-b[i+2]); return total / (32*32*3*255); }
function time(t) { const m = Math.floor(t/60), s = Math.floor(t%60); return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`; }
function addCaption(text, timestamp) {
  if (ui.output.querySelector('.output-empty')) ui.output.textContent = '';
  const row = document.createElement('div'); row.className = 'caption-row';
  const stamp = document.createElement('span'); stamp.className='caption-time'; stamp.textContent = timestamp == null ? 'IMAGE' : time(timestamp);
  const caption = document.createElement('p'); caption.textContent = text; row.append(stamp, caption); ui.output.prepend(row);
}
async function generate(element, timestamp, force = false) {
  if (busy || !ready) return;
  const currentSession = session;
  try {
    const sample = frame(element);
    if (!force && previousFrame && difference(previousFrame, sample.pixels) < Number(ui.threshold.value)) return;
    busy = true; controls(); ui['result-status'].textContent = 'GENERATING';
    const result = await post('/api/caption', {image:sample.image, prompt:ui.prompt.value, max_tokens:100});
    if (currentSession !== session) return;
    previousFrame = sample.pixels;
    if (result.caption && (timestamp == null || result.caption !== lastCaption)) { addCaption(result.caption, timestamp); lastCaption = result.caption; }
    ui['result-status'].textContent = timestamp == null ? 'COMPLETE' : 'LIVE CAPTIONS';
  } catch (e) { if (currentSession === session) { ui['result-status'].textContent = 'ERROR'; addCaption(e.message, timestamp); if (mediaType === 'video') ui.video.pause(); } }
  finally { busy = false; controls(); }
}
ui.run.onclick = async () => {
  if (mediaType === 'image') return generate(ui.image, null, true);
  if (ui.video.ended) ui.video.currentTime = 0;
  previousFrame = null; lastSample = -Infinity;
  try { await ui.video.play(); } catch (e) { ui['result-status'].textContent = e.message; }
};
ui.video.addEventListener('seeked', () => { previousFrame = null; lastSample = -Infinity; });
ui.video.addEventListener('pause', () => { if (mediaType === 'video' && !busy) ui['result-status'].textContent = ui.video.ended ? 'COMPLETE' : 'PAUSED'; });
setInterval(() => {
  if (mediaType !== 'video' || ui.video.paused || ui.video.seeking || busy || !ready) return;
  const interval = Math.max(.5, Math.min(30, Number(ui.interval.value) || 2));
  if (ui.video.currentTime - lastSample >= interval) { lastSample = ui.video.currentTime; generate(ui.video, lastSample); }
}, 250);
setInterval(status, 2000); status();
