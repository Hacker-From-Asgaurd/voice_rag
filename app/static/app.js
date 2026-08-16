document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const micButton = document.getElementById('micButton');
  const micStatusText = document.getElementById('micStatusText');
  const micSubtext = document.getElementById('micSubtext');
  const pulseRing = document.getElementById('pulseRing');
  const waveformCanvas = document.getElementById('waveformCanvas');
  const transcriptText = document.getElementById('transcriptText');
  const detectedLangTag = document.getElementById('detectedLangTag');
  const answerBody = document.getElementById('answerBody');
  const answerMeta = document.getElementById('answerMeta');
  const groundingBadge = document.getElementById('groundingBadge');
  const groundingBadgeText = document.getElementById('groundingBadgeText');
  const sourceCount = document.getElementById('sourceCount');
  const sourcesList = document.getElementById('sourcesList');

  // Latency elements
  const liveCoreVal = document.getElementById('liveCoreVal');
  const benchmarkP95Val = document.getElementById('benchmarkP95Val');
  const wfRowStt = document.getElementById('wfRowStt');
  const wfSttTime = document.getElementById('wfSttTime');
  const wfGuardrailTime = document.getElementById('wfGuardrailTime');
  const wfRetrievalTime = document.getElementById('wfRetrievalTime');
  const wfRerankTime = document.getElementById('wfRerankTime');
  const wfGateTime = document.getElementById('wfGateTime');
  const wfGenTime = document.getElementById('wfGenTime');
  const wfTotalTime = document.getElementById('wfTotalTime');

  const barStt = document.getElementById('barStt');
  const barGuardrail = document.getElementById('barGuardrail');
  const barRetrieval = document.getElementById('barRetrieval');
  const barRerank = document.getElementById('barRerank');
  const barGate = document.getElementById('barGate');
  const barGen = document.getElementById('barGen');
  const barTotal = document.getElementById('barTotal');

  // Text input fallback elements
  const toggleTextInputBtn = document.getElementById('toggleTextInputBtn');
  const textInputBox = document.getElementById('textInputBox');
  const textQueryInput = document.getElementById('textQueryInput');
  const submitTextBtn = document.getElementById('submitTextBtn');

  // Audio Recording State
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let audioContext = null;
  let analyser = null;
  let animationFrameId = null;

  // 1. Fetch System Benchmark Metrics on Startup
  fetchBenchmarkMetrics();

  // 2. Setup Event Listeners
  micButton.addEventListener('click', toggleRecording);

  toggleTextInputBtn.addEventListener('click', () => {
    textInputBox.classList.toggle('hidden');
    if (!textInputBox.classList.contains('hidden')) {
      textQueryInput.focus();
    }
  });

  submitTextBtn.addEventListener('click', handleTextQuery);
  textQueryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleTextQuery();
  });

  // -------------------------------------------------------------
  // MICROPHONE & AUDIO RECORDING FLOW
  // -------------------------------------------------------------
  async function toggleRecording() {
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  }

  async function startRecording() {
    audioChunks = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Setup Web Audio Analyser for Waveform
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);
      drawWaveform();

      // Preferred MediaRecorder format
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      mediaRecorder = new MediaRecorder(stream, { mimeType });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        cancelAnimationFrame(animationFrameId);
        clearCanvas();
        if (audioContext) audioContext.close();

        // Release mic tracks
        stream.getTracks().forEach(track => track.stop());

        const audioBlob = new Blob(audioChunks, { type: mimeType });
        if (audioBlob.size > 500) {
          await sendVoiceQuery(audioBlob);
        } else {
          showError("Audio recording was too short.");
          resetMicUI();
        }
      };

      mediaRecorder.start();
      isRecording = true;

      // Update UI to RECORDING state
      micButton.classList.add('recording');
      pulseRing.classList.add('recording');
      micStatusText.textContent = "Listening...";
      micSubtext.textContent = "Tap again when finished speaking";
      detectedLangTag.textContent = "Recording...";
      transcriptText.textContent = "Capturing voice stream...";
    } catch (err) {
      console.error("Microphone access error:", err);
      showError("Microphone permission denied or device not found.");
      resetMicUI();
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      isRecording = false;

      // Update UI to PROCESSING state
      micButton.classList.remove('recording');
      pulseRing.classList.remove('recording');
      micButton.classList.add('processing');
      micStatusText.textContent = "Processing...";
      micSubtext.textContent = "Transcribing with Sarvam Saaras v3 & searching E5...";
    }
  }

  function resetMicUI() {
    isRecording = false;
    micButton.classList.remove('recording', 'processing');
    pulseRing.classList.remove('recording');
    micStatusText.textContent = "Tap to speak";
    micSubtext.textContent = "Hindi · English · Marathi · Code-Mixed";
  }

  // -------------------------------------------------------------
  // WAVEFORM VISUALIZER
  // -------------------------------------------------------------
  function drawWaveform() {
    if (!analyser) return;
    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function render() {
      animationFrameId = requestAnimationFrame(render);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const barWidth = (canvas.width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        ctx.fillStyle = `rgba(244, 63, 94, ${0.4 + (dataArray[i] / 255) * 0.6})`;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
      }
    }
    render();
  }

  function clearCanvas() {
    const ctx = waveformCanvas.getContext('2d');
    ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  }

  // -------------------------------------------------------------
  // API DISPATCH: VOICE QUERY
  // -------------------------------------------------------------
  async function sendVoiceQuery(audioBlob) {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');

    try {
      const res = await fetch('/api/voice-query', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      renderResponse(data, true);
    } catch (err) {
      console.error("Voice query failed:", err);
      showError(`Voice query failed: ${err.message}`);
    } finally {
      resetMicUI();
    }
  }

  // -------------------------------------------------------------
  // API DISPATCH: TEXT QUERY
  // -------------------------------------------------------------
  async function handleTextQuery() {
    const query = textQueryInput.value.trim();
    if (!query) return;

    micStatusText.textContent = "Processing text...";
    submitTextBtn.disabled = true;
    submitTextBtn.textContent = "...";

    try {
      const res = await fetch('/api/text-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      renderResponse(data, false);
    } catch (err) {
      console.error("Text query failed:", err);
      showError(`Text query failed: ${err.message}`);
    } finally {
      submitTextBtn.disabled = false;
      submitTextBtn.textContent = "Search";
      resetMicUI();
    }
  }

  // -------------------------------------------------------------
  // RESPONSE RENDERING
  // -------------------------------------------------------------
  function renderResponse(data, isVoice) {
    // 1. Transcript & Language
    if (isVoice) {
      transcriptText.textContent = data.transcript || "[Empty Transcript]";
      detectedLangTag.textContent = data.language_code || "Unknown";
    } else {
      transcriptText.textContent = data.query;
      detectedLangTag.textContent = "Text Query";
    }

    // 2. Answer & Grounding Status Badge
    answerBody.innerHTML = `<p>${escapeHtml(data.answer)}</p>`;

    groundingBadge.className = 'grounding-badge';
    if (data.answer.includes("सहायता नहीं कर सकता") || data.answer.includes("इस तरह के अनुरोध")) {
      groundingBadge.classList.add('blocked');
      groundingBadgeText.textContent = 'BLOCKED';
      answerMeta.textContent = "Request blocked by safety guardrail · 0 sources dispatched";
    } else if (data.grounded) {
      groundingBadge.classList.add('grounded');
      groundingBadgeText.textContent = 'GROUNDED';
      answerMeta.textContent = `Evidence-backed response · ${(data.sources || []).length} retrieved sources · Candidate pool k=15`;
    } else {
      groundingBadge.classList.add('abstained');
      groundingBadgeText.textContent = 'SAFE ABSTENTION';
      answerMeta.textContent = "Safe abstention triggered · 0 evidence sources passed gate";
    }

    // 3. Source Attribution Cards
    renderSources(data.sources || []);

    // 4. Latency Waterfall
    renderLatencyWaterfall(data.latency || {}, isVoice);
  }

  function renderSources(sources) {
    sourceCount.textContent = `${sources.length} retrieved passages`;
    if (!sources || sources.length === 0) {
      sourcesList.innerHTML = '<div class="empty-sources">No supporting passages retained by evidence gate.</div>';
      return;
    }

    sourcesList.innerHTML = sources.map((s, idx) => {
      const srcNum = String(idx + 1).padStart(2, '0');
      return `
        <div class="source-item" id="sourceCard_${idx}">
          <div class="source-header">
            <span class="source-title">SOURCE ${srcNum}</span>
            <div class="source-metrics">
              <span class="source-metric-pill">Query #${s.query_id}</span>
              <span class="source-metric-pill">Passage #${s.passage_id}</span>
              <span class="source-metric-pill">Score: ${s.score.toFixed(2)}</span>
            </div>
          </div>
          <div class="source-snippet" id="sourceSnippet_${idx}">${escapeHtml(s.chunk)}</div>
          <button class="source-expand-btn" onclick="toggleSourceExpand(${idx})">View full evidence ▾</button>
        </div>
      `;
    }).join('');
  }

  window.toggleSourceExpand = function(idx) {
    const card = document.getElementById(`sourceCard_${idx}`);
    if (!card) return;
    card.classList.toggle('expanded');
    const btn = card.querySelector('.source-expand-btn');
    if (btn) {
      btn.textContent = card.classList.contains('expanded') ? 'Hide evidence ▴' : 'View full evidence ▾';
    }
  };

  function renderLatencyWaterfall(lat, isVoice) {
    const stt = isVoice ? (lat.stt_ms || 0) : 0;
    const gr = lat.guardrail_ms || 0;
    const ret = lat.retrieval_ms || 0;
    const rerank = lat.reranker_ms || 0;
    const gate = lat.evidence_gate_ms || 0;
    const gen = lat.generation_ms || 0;
    const total = lat.total_ms || (stt + gr + ret + rerank + gate + gen);

    // Live Retrieval Core calculation (Single-request measurement)
    const core = ret + rerank;
    if (liveCoreVal) {
      liveCoreVal.textContent = `${core.toFixed(1)} ms`;
    }

    // Toggle STT row visibility
    if (wfRowStt) {
      wfRowStt.style.display = isVoice ? 'flex' : 'none';
    }

    wfSttTime.textContent = `${stt.toFixed(1)} ms`;
    wfGuardrailTime.textContent = `${gr.toFixed(1)} ms`;
    wfRetrievalTime.textContent = `${ret.toFixed(1)} ms`;
    wfRerankTime.textContent = `${rerank.toFixed(1)} ms`;
    wfGateTime.textContent = `${gate.toFixed(1)} ms`;
    wfGenTime.textContent = `${gen.toFixed(1)} ms`;
    wfTotalTime.textContent = `${total.toFixed(1)} ms`;

    const scale = total > 0 ? total : 1;
    barStt.style.width = `${Math.min(100, (stt / scale) * 100)}%`;
    barGuardrail.style.width = `${Math.min(100, Math.max(0.5, (gr / scale) * 100))}%`;
    barRetrieval.style.width = `${Math.min(100, (ret / scale) * 100)}%`;
    barRerank.style.width = `${Math.min(100, (rerank / scale) * 100)}%`;
    barGate.style.width = `${Math.min(100, Math.max(0.5, (gate / scale) * 100))}%`;
    barGen.style.width = `${Math.min(100, (gen / scale) * 100)}%`;
    barTotal.style.width = '100%';
  }

  // -------------------------------------------------------------
  // BENCHMARK STATS FETCH
  // -------------------------------------------------------------
  async function fetchBenchmarkMetrics() {
    try {
      const res = await fetch('/api/metrics');
      if (res.ok) {
        const m = await res.json();
        document.getElementById('bmRecall1').textContent = `${m.recall_at_1.toFixed(2)}%`;
        document.getElementById('bmRecall5').textContent = `${m.recall_at_5.toFixed(2)}%`;
        document.getElementById('bmMrr').textContent = m.mrr.toFixed(4);
        document.getElementById('bmP50').textContent = `${m.p50_ms.toFixed(1)} ms`;
        document.getElementById('bmP95').textContent = `${m.p95_ms.toFixed(1)} ms`;
        document.getElementById('bmP100').textContent = `${m.p100_ms.toFixed(1)} ms`;
        if (benchmarkP95Val) {
          benchmarkP95Val.textContent = `P95: ${m.p95_ms.toFixed(1)} ms`;
        }
      }
    } catch (e) {
      console.warn("Could not fetch metrics:", e);
    }
  }

  function showError(msg) {
    answerBody.innerHTML = `<p style="color: var(--accent-rose);">${escapeHtml(msg)}</p>`;
    groundingBadge.className = 'grounding-badge blocked';
    groundingBadgeText.textContent = 'ERROR';
    answerMeta.textContent = "Pipeline execution error";
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});
