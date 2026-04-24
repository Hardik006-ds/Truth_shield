# app.py
# TRUTH SHIELD - Binary Decision System (AI vs REAL only)
# Version: 3.0.0

import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
import json
from datetime import datetime

# ==============================================================================
# APPLICATION SETUP
# ==============================================================================

app = FastAPI(
    title="Truth Shield",
    description="Binary AI Detection - Real or AI Only",
    version="3.0.0"
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

FEEDBACK_FILE = BASE_DIR / "feedback.json"
if not FEEDBACK_FILE.exists():
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump({"correct": [], "incorrect": []}, f)

# ==============================================================================
# ML MODELS (Your trained models)
# ==============================================================================

from models.detector import ContentDetector
from models.text_detector import TextDetector

content_detector = ContentDetector()
text_detector = TextDetector()

# ==============================================================================
# FRONTEND - BINARY UI (No Uncertain State)
# ==============================================================================

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TRUTH SHIELD // BINARY DETECTION</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Orbitron:wght@700;900&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --void: #050505;
            --cyan: #00f0ff;
            --red: #ff003c;      /* AI Color */
            --green: #00ff88;    /* REAL Color */
            --white: #ffffff;
        }
        
        body {
            background: var(--void);
            color: white;
            font-family: 'JetBrains Mono', monospace;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* CRT Scanlines */
        body::before {
            content: "";
            position: fixed;
            inset: 0;
            background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.1) 2px, rgba(0,0,0,0.1) 4px);
            pointer-events: none;
            z-index: 9999;
            animation: scanline 8s linear infinite;
        }
        
        @keyframes scanline {
            0% { transform: translateY(0); }
            100% { transform: translateY(100vh); }
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            position: relative;
            z-index: 10;
        }
        
        /* Header */
        .header {
            text-align: center;
            padding: 3rem 0;
            border-bottom: 1px solid rgba(0,240,255,0.2);
            margin-bottom: 3rem;
        }
        
        .glitch-title {
            font-family: 'Orbitron', sans-serif;
            font-size: clamp(2.5rem, 8vw, 4rem);
            font-weight: 900;
            color: var(--cyan);
            text-shadow: 0 0 20px var(--cyan);
            position: relative;
            display: inline-block;
            letter-spacing: 0.1em;
        }
        
        .glitch-title::before, .glitch-title::after {
            content: 'TRUTH SHIELD';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        
        .glitch-title::before {
            left: 2px;
            text-shadow: -2px 0 var(--red);
            clip: rect(24px, 450px, 56px, 0);
            animation: glitch1 5s infinite linear alternate-reverse;
        }
        
        .glitch-title::after {
            left: -2px;
            text-shadow: -2px 0 var(--cyan);
            clip: rect(44px, 450px, 56px, 0);
            animation: glitch2 5s infinite linear alternate-reverse;
        }
        
        @keyframes glitch1 {
            0% { clip: rect(30px, 9999px, 50px, 0); }
            20% { clip: rect(80px, 9999px, 10px, 0); }
            40% { clip: rect(10px, 9999px, 80px, 0); }
            60% { clip: rect(60px, 9999px, 15px, 0); }
            80% { clip: rect(30px, 9999px, 90px, 0); }
        }
        
        @keyframes glitch2 {
            0% { clip: rect(60px, 9999px, 100px, 0); }
            20% { clip: rect(10px, 9999px, 50px, 0); }
            40% { clip: rect(90px, 9999px, 20px, 0); }
            60% { clip: rect(40px, 9999px, 70px, 0); }
            80% { clip: rect(70px, 9999px, 10px, 0); }
        }
        
        .subtitle {
            color: rgba(255,255,255,0.6);
            margin-top: 1rem;
            font-size: 0.9rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
        }
        
        /* Mode Selector */
        .mode-selector {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            justify-content: center;
        }
        
        .mode-btn {
            flex: 1;
            padding: 1rem;
            background: rgba(0,0,0,0.5);
            border: 1px solid rgba(0,240,255,0.3);
            color: rgba(255,255,255,0.6);
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            font-family: inherit;
            font-weight: 700;
            letter-spacing: 0.1em;
            border-radius: 8px;
            max-width: 200px;
        }
        
        .mode-btn.active {
            background: var(--cyan);
            color: var(--void);
            box-shadow: 0 0 20px var(--cyan);
        }
        
        .mode-btn:hover:not(.active) {
            border-color: var(--cyan);
            color: var(--cyan);
        }
        
        /* Upload Zone */
        .upload-area {
            border: 2px dashed rgba(0,240,255,0.3);
            border-radius: 12px;
            padding: 4rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.4s;
            margin-bottom: 2rem;
            background: rgba(0,0,0,0.3);
        }
        
        .upload-area:hover {
            border-color: var(--cyan);
            background: rgba(0,240,255,0.05);
            box-shadow: 0 0 30px rgba(0,240,255,0.2);
        }
        
        .upload-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        /* File Preview */
        .file-preview {
            display: none;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: rgba(0,0,0,0.5);
            border: 1px solid var(--cyan);
            border-radius: 8px;
            margin-bottom: 2rem;
            animation: slideIn 0.3s;
        }
        
        .file-preview.show { display: flex; }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .file-preview img {
            width: 60px;
            height: 60px;
            object-fit: cover;
            border-radius: 4px;
            filter: grayscale(100%) brightness(0.8);
        }
        
        .file-info { flex: 1; text-align: left; }
        .file-name { color: var(--cyan); font-weight: 700; }
        .file-size { color: rgba(255,255,255,0.5); font-size: 0.9rem; }
        
        .remove-btn {
            background: var(--red);
            border: none;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Text Input */
        .text-input {
            display: none;
            margin-bottom: 2rem;
        }
        
        .text-input.active { display: block; }
        
        textarea {
            width: 100%;
            min-height: 200px;
            padding: 1.5rem;
            background: rgba(0,0,0,0.5);
            border: 1px solid rgba(0,240,255,0.3);
            color: var(--cyan);
            font-family: inherit;
            font-size: 1rem;
            border-radius: 8px;
            resize: vertical;
        }
        
        textarea:focus {
            outline: none;
            border-color: var(--cyan);
            box-shadow: 0 0 20px rgba(0,240,255,0.2);
        }
        
        /* Scan Button */
        .scan-btn {
            width: 100%;
            padding: 1.25rem;
            background: linear-gradient(135deg, var(--red), #cc0030);
            border: none;
            border-radius: 8px;
            color: white;
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 20px rgba(255,0,60,0.3);
            margin-bottom: 2rem;
        }
        
        .scan-btn:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(255,0,60,0.5);
        }
        
        .scan-btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }
        
        .scan-btn.loading {
            background: linear-gradient(90deg, var(--cyan), var(--cyan));
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 4px 20px var(--cyan); }
            50% { box-shadow: 0 4px 50px var(--cyan), 0 0 80px var(--cyan); }
        }
        
        /* RESULTS PANEL - BINARY ONLY */
        .results {
            display: none;
            margin-top: 2rem;
            padding: 3rem;
            background: rgba(0,0,0,0.9);
            border: 2px solid;
            border-radius: 16px;
            text-align: center;
            animation: fadeUp 0.5s;
        }
        
        .results.show { display: block; }
        
        /* AI State - Red */
        .results.ai {
            border-color: var(--red);
            box-shadow: 0 0 60px rgba(255,0,60,0.3);
        }
        
        /* REAL State - Green */
        .results.real {
            border-color: var(--green);
            box-shadow: 0 0 60px rgba(0,255,136,0.3);
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .big-verdict {
            font-size: 3rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 2rem;
            padding: 1.5rem 3rem;
            display: inline-block;
            border-radius: 12px;
            animation: badgePop 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        
        @keyframes badgePop {
            0% { transform: scale(0); }
            80% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        .big-verdict.ai {
            background: rgba(255,0,60,0.15);
            border: 2px solid var(--red);
            color: var(--red);
            box-shadow: 0 0 40px rgba(255,0,60,0.4);
        }
        
        .big-verdict.real {
            background: rgba(0,255,136,0.15);
            border: 2px solid var(--green);
            color: var(--green);
            box-shadow: 0 0 40px rgba(0,255,136,0.4);
        }
        
        .score-display {
            margin: 2rem 0;
        }
        
        .score-value {
            font-size: 4rem;
            font-weight: 900;
            font-family: 'Orbitron', sans-serif;
        }
        
        .score-value.ai { color: var(--red); text-shadow: 0 0 20px var(--red); }
        .score-value.real { color: var(--green); text-shadow: 0 0 20px var(--green); }
        
        .score-label {
            font-size: 1rem;
            color: rgba(255,255,255,0.6);
            margin-top: 0.5rem;
        }
        
        .verdict-text {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 1rem 0;
            text-transform: uppercase;
        }
        
        .verdict-text.ai { color: var(--red); }
        .verdict-text.real { color: var(--green); }
        
        .confidence-section {
            margin-top: 2rem;
        }
        
        .conf-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            color: rgba(255,255,255,0.7);
        }
        
        .conf-bar {
            height: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
            overflow: hidden;
        }
        
        .conf-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 1s ease-out;
            width: 0%;
        }
        
        .conf-fill.ai { background: linear-gradient(90deg, var(--red), #ff4d7a); }
        .conf-fill.real { background: linear-gradient(90deg, var(--green), #40ffa0); }
        
        /* Feedback */
        .feedback {
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .feedback-text {
            color: rgba(255,255,255,0.6);
            margin-bottom: 1rem;
        }
        
        .feedback-btns {
            display: flex;
            gap: 1rem;
            justify-content: center;
        }
        
        .feedback-btn {
            padding: 0.75rem 2rem;
            border: 2px solid;
            background: transparent;
            font-family: inherit;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .feedback-btn.yes {
            border-color: var(--green);
            color: var(--green);
        }
        
        .feedback-btn.yes:hover {
            background: var(--green);
            color: var(--void);
            box-shadow: 0 0 20px var(--green);
        }
        
        .feedback-btn.no {
            border-color: var(--red);
            color: var(--red);
        }
        
        .feedback-btn.no:hover {
            background: var(--red);
            color: white;
            box-shadow: 0 0 20px var(--red);
        }
        
        input[type="file"] { display: none; }
        
        @media (max-width: 768px) {
            .big-verdict { font-size: 2rem; padding: 1rem 2rem; }
            .score-value { font-size: 3rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="glitch-title">TRUTH SHIELD</div>
            <div class="subtitle">Binary AI Detection Protocol</div>
        </div>
        
        <div class="mode-selector">
            <button class="mode-btn active" onclick="setMode('image')">Image</button>
            <button class="mode-btn" onclick="setMode('text')">Text</button>
        </div>
        
        <div id="image-section">
            <div class="upload-area" onclick="document.getElementById('file-input').click()">
                <div class="upload-icon">◉</div>
                <div style="font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--cyan); margin-bottom: 0.5rem;">
                    Drop Signal / Click to Upload
                </div>
                <div style="color: rgba(0,240,255,0.5); font-size: 0.9rem;">
                    PNG, JPG, JPEG • Max 10MB
                </div>
            </div>
            <input type="file" id="file-input" accept="image/*" onchange="handleFile(this)">
            
            <div class="file-preview" id="file-preview">
                <img id="preview-img" src="" alt="">
                <div class="file-info">
                    <div class="file-name" id="file-name">signal.dat</div>
                    <div class="file-size" id="file-size">0 KB</div>
                </div>
                <button class="remove-btn" onclick="removeFile(event)">×</button>
            </div>
        </div>
        
        <div class="text-input" id="text-section">
            <textarea id="text-content" placeholder="// Paste text for AI detection..." oninput="updateCharCount()"></textarea>
            <div style="text-align: right; margin-top: 0.5rem; color: rgba(255,255,255,0.4); font-size: 0.9rem;">
                <span id="char-count">0</span> characters
            </div>
        </div>
        
        <button class="scan-btn" id="scan-btn" onclick="initiateScan()" disabled>
            Initiate Binary Scan
        </button>
        
        <div class="results" id="results">
            <div class="big-verdict ai" id="big-verdict">
                ANALYZING...
            </div>
            
            <div class="score-display">
                <div class="score-value" id="score-display-value">0.0</div>
                <div class="score-label">AI Probability Score / 10</div>
            </div>
            
            <div class="verdict-text" id="verdict-text">Processing...</div>
            
            <div class="confidence-section">
                <div class="conf-header">
                    <span>Confidence Level</span>
                    <span id="conf-percentage">0%</span>
                </div>
                <div class="conf-bar">
                    <div class="conf-fill" id="conf-fill"></div>
                </div>
            </div>
            
            <div class="feedback">
                <div class="feedback-text">Was this analysis correct?</div>
                <div class="feedback-btns">
                    <button class="feedback-btn yes" onclick="submitFeedback(true)">✓ YES</button>
                    <button class="feedback-btn no" onclick="submitFeedback(false)">✗ NO</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // State
        let currentMode = 'image';
        let selectedFile = null;
        let lastResult = null;
        let isScanning = false;

        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-btn').forEach((t, i) => {
                t.classList.remove('active');
                if ((mode === 'image' && i === 0) || (mode === 'text' && i === 1)) {
                    t.classList.add('active');
                }
            });

            document.getElementById('image-section').style.display = mode === 'image' ? 'block' : 'none';
            document.getElementById('text-section').classList.toggle('active', mode === 'text');
            
            if (mode === 'image') {
                document.getElementById('text-content').value = '';
            } else {
                selectedFile = null;
                document.getElementById('file-input').value = '';
                document.getElementById('file-preview').classList.remove('show');
            }
            updateScanButton();
        }

        function handleFile(input) {
            const file = input.files[0];
            if (!file) return;
            if (file.size > 10 * 1024 * 1024) {
                alert('File too large. Maximum is 10MB.');
                return;
            }
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('preview-img').src = e.target.result;
            };
            reader.readAsDataURL(file);
            document.getElementById('file-name').textContent = file.name;
            document.getElementById('file-size').textContent = formatSize(file.size);
            document.getElementById('file-preview').classList.add('show');
            updateScanButton();
        }

        function removeFile(e) {
            e.stopPropagation();
            selectedFile = null;
            document.getElementById('file-input').value = '';
            document.getElementById('file-preview').classList.remove('show');
            updateScanButton();
        }

        function formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }

        function updateCharCount() {
            const len = document.getElementById('text-content').value.length;
            document.getElementById('char-count').textContent = len;
            updateScanButton();
        }

        function updateScanButton() {
            const btn = document.getElementById('scan-btn');
            let valid = false;
            if (currentMode === 'image') {
                valid = selectedFile !== null;
            } else {
                valid = document.getElementById('text-content').value.trim().length >= 10;
            }
            btn.disabled = !valid;
        }

        async function initiateScan() {
            if (isScanning) return;
            isScanning = true;
            
            const btn = document.getElementById('scan-btn');
            const results = document.getElementById('results');
            
            btn.classList.add('loading');
            btn.innerHTML = 'SCANNING...';
            results.classList.remove('show');

            const formData = new FormData();
            formData.append('content_type', currentMode);
            
            if (currentMode === 'image') {
                formData.append('file', selectedFile);
            } else {
                formData.append('text', document.getElementById('text-content').value);
            }

            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.error) throw new Error(data.error);
                
                lastResult = data;
                displayResults(data);
                
            } catch (error) {
                console.error('Error:', error);
                alert('Analysis failed: ' + error.message);
                isScanning = false;
                btn.classList.remove('loading');
                btn.innerHTML = 'Initiate Binary Scan';
            }
        }

        // BINARY DISPLAY FUNCTION - NO UNCERTAIN
        function displayResults(data) {
            isScanning = false;
            const btn = document.getElementById('scan-btn');
            btn.classList.remove('loading');
            btn.innerHTML = 'Initiate Binary Scan';

            const score = data.score;
            const verdict = data.verdict;
            
            // BINARY DECISION: >= 5 = AI, < 5 = REAL
            const isAI = score >= 5.0;
            
            // Elements
            const results = document.getElementById('results');
            const bigVerdict = document.getElementById('big-verdict');
            const scoreValue = document.getElementById('score-display-value');
            const verdictText = document.getElementById('verdict-text');
            const confFill = document.getElementById('conf-fill');
            const confPercentage = document.getElementById('conf-percentage');
            
            // Reset classes
            results.classList.remove('ai', 'real');
            scoreValue.classList.remove('ai', 'real');
            verdictText.classList.remove('ai', 'real');
            confFill.classList.remove('ai', 'real');
            
            if (isAI) {
                // AI DETECTED (Red)
                results.classList.add('ai');
                scoreValue.classList.add('ai');
                verdictText.classList.add('ai');
                confFill.classList.add('ai');
                
                bigVerdict.className = 'big-verdict ai';
                bigVerdict.textContent = '⚠️ AI DETECTED';
                
                verdictText.textContent = 'AI';
                scoreValue.textContent = score.toFixed(1);
                
                // Confidence: 50% at 5.0, 100% at 10.0
                const confidencePct = 50 + ((score - 5) / 5) * 50;
                confFill.style.width = confidencePct + '%';
                confPercentage.textContent = Math.round(confidencePct) + '%';
                
            } else {
                // AUTHENTIC/REAL (Green)
                results.classList.add('real');
                scoreValue.classList.add('real');
                verdictText.classList.add('real');
                confFill.classList.add('real');
                
                bigVerdict.className = 'big-verdict real';
                bigVerdict.textContent = '✓ AUTHENTIC';
                
                verdictText.textContent = 'REAL';
                scoreValue.textContent = score.toFixed(1);
                
                // Confidence: 50% at 5.0, 100% at 0.0
                const confidencePct = 50 + ((5 - score) / 5) * 50;
                confFill.style.width = confidencePct + '%';
                confPercentage.textContent = Math.round(confidencePct) + '%';
            }
            
            // Show results
            results.classList.add('show');
        }

        async function submitFeedback(isCorrect) {
            if (!lastResult) return;
            
            const btns = document.querySelectorAll('.feedback-btn');
            btns.forEach(b => b.style.pointerEvents = 'none');
            
            const formData = new FormData();
            formData.append('is_correct', isCorrect.toString());
            formData.append('content_type', currentMode);
            formData.append('score', lastResult.score);
            formData.append('verdict', lastResult.verdict);
            
            try {
                await fetch('/feedback', { method: 'POST', body: formData });
                
                setTimeout(() => {
                    document.getElementById('results').classList.remove('show');
                    if (currentMode === 'image') {
                        removeFile({ stopPropagation: () => {} });
                    } else {
                        document.getElementById('text-content').value = '';
                        updateCharCount();
                    }
                    btns.forEach(b => b.style.pointerEvents = 'auto');
                }, 2000);
                
            } catch (e) {
                console.error(e);
                btns.forEach(b => b.style.pointerEvents = 'auto');
            }
        }

        // Drag and drop
        const uploadZone = document.querySelector('.upload-area');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
            uploadZone.addEventListener(e, (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
            }, false);
        });
        
        ['dragenter', 'dragover'].forEach(e => {
            uploadZone.addEventListener(e, () => uploadZone.classList.add('drag-active'), false);
        });
        
        ['dragleave', 'drop'].forEach(e => {
            uploadZone.addEventListener(e, () => uploadZone.classList.remove('drag-active'), false);
        });
        
        uploadZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length) handleFile({ files: files });
        }, false);

        updateScanButton();
    </script>
</body>
</html>"""

# ==============================================================================
# BACKEND API (Unchanged logic)
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_CONTENT

@app.post("/analyze")
async def analyze_content(
    content_type: str = Form(...),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    try:
        if content_type == "image":
            if not file:
                raise HTTPException(status_code=400, detail="No file provided")
            
            ext = (file.filename or "img.jpg").split(".")[-1].lower()
            if ext not in ['jpg', 'jpeg', 'png']:
                raise HTTPException(status_code=400, detail="Unsupported format")
            
            filepath = UPLOAD_DIR / f"{uuid.uuid4()}.{ext}"
            content = await file.read()
            with open(filepath, "wb") as f:
                f.write(content)
            
            try:
                result = await content_detector.analyze_image(filepath)
                return JSONResponse(content=result)
            finally:
                filepath.unlink(missing_ok=True)
                
        elif content_type == "text":
            if not text or len(text.strip()) < 10:
                raise HTTPException(status_code=400, detail="Text too short")
            
            result = text_detector.analyze(text)
            return JSONResponse(content=result)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid type")
            
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/feedback")
async def submit_feedback(
    is_correct: str = Form(...),
    content_type: str = Form(...),
    score: float = Form(...),
    verdict: str = Form(...)
):
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            data = json.load(f)
        
        entry = {
            "timestamp": str(datetime.now()),
            "content_type": content_type,
            "predicted_score": score,
            "predicted_verdict": verdict,
            "was_correct": is_correct.lower() == 'true',
            "id": str(uuid.uuid4())[:8]
        }
        
        if entry["was_correct"]:
            data["correct"].append(entry)
        else:
            data["incorrect"].append(entry)
        
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        return JSONResponse({
            "status": "recorded",
            "correct": len(data["correct"]),
            "incorrect": len(data["incorrect"])
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "3.0.0", "mode": "binary"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
