const WS_URL = 'ws://127.0.0.1:8000/ws';
let ws = null;
let currentTracks = [];

const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const ctx = canvas.getContext('2d');

// 1. WebSocket Setup
function connectWebSocket() {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => console.log("Connected to AI Backend");
    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'tracks') {
            currentTracks = msg.tracks;
            updateUIList();
        }
    };
    // Reconnect if closed
    ws.onclose = () => setTimeout(connectWebSocket, 2000);
}

// 2. Automated Camera Start
async function init() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 1280, height: 720, frameRate: 30 } 
        });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            connectWebSocket();
            drawLoop();
            logicLoop();
        };
    } catch (err) {
        console.error("Camera access denied or not found:", err);
        alert("Please allow camera access for the tracker to work.");
    }
}

// 3. Visualization Loop (Smooth 60fps)
function drawLoop() {
    // Sync canvas size to displayed video size
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    currentTracks.forEach(track => {
        // Defensive check: ensure bbox exists
        if (!track.bbox) return;

        const x = track.bbox.x * canvas.width;
        const y = track.bbox.y * canvas.height;
        const w = track.bbox.w * canvas.width;
        const h = track.bbox.h * canvas.height;

        // --- Improved Color Logic for Retries ---
        let boxColor = '#00FF00'; // Default Green (Matched)
        let labelBgColor = 'rgba(0, 255, 0, 0.8)';
        
        if (track.name.includes('Matching')) {
            boxColor = '#FFFF00'; // Yellow while retrying
            labelBgColor = 'rgba(255, 255, 0, 0.8)';
        } else if (track.name.includes('Unknown') || track.name === 'Detecting...') {
            boxColor = '#FF6600'; // Orange
            labelBgColor = 'rgba(255, 102, 0, 0.8)';
        } else if (track.confidence > 0 && track.confidence < 75) {
            boxColor = '#FF9900'; // Gold for low confidence
            labelBgColor = 'rgba(255, 153, 0, 0.8)';
        }

        // 1. Draw Bounding Box
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // 2. Draw Centroid (with fallback if missing)
        if (track.centroid) {
            ctx.fillStyle = '#FF0000';
            ctx.beginPath();
            ctx.arc(track.centroid.x * canvas.width, track.centroid.y * canvas.height, 4, 0, Math.PI * 2);
            ctx.fill();
        }

        // 3. Label Text and Background
        const label = track.name || `ID ${track.trackId}`;
        ctx.font = 'bold 16px Arial';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = 20;
        
        // Ensure label stays on screen
        const labelX = x;
        const labelY = Math.max(y - 5, textHeight + 5);
        
        // Label Background
        ctx.fillStyle = labelBgColor;
        ctx.fillRect(labelX, labelY - textHeight - 2, textWidth + 12, textHeight + 8);
        
        // Label Text
        ctx.fillStyle = '#000000';
        ctx.fillText(label, labelX + 6, labelY - 6);
        
        // 4. Track ID Badge (inside the box)
        ctx.font = 'bold 12px Arial';
        ctx.fillStyle = 'white';
        ctx.shadowColor = "black";
        ctx.shadowBlur = 4;
        ctx.fillText(`#${track.trackId}`, x + 5, y + 15);
        ctx.shadowBlur = 0; // Reset shadow
    });

    requestAnimationFrame(drawLoop);
}

// 4. Processing Loop (Sending frames to server)
async function logicLoop() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 480; 
        tempCanvas.height = 270;
        
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 480, 270);
        
        const data = tempCanvas.toDataURL('image/jpeg', 0.5);
        ws.send(JSON.stringify({ type: 'frame', data }));
    }
    setTimeout(logicLoop, 250);  // Increased from 150ms to 250ms
}

function updateUIList() {
    const list = document.getElementById('cardsList');
    
    if (currentTracks.length === 0) {
        list.innerHTML = '<div style="color: #64748b; text-align: center; margin-top: 2rem;">No cards detected</div>';
        return;
    }
    
    list.innerHTML = currentTracks.map(t => {
        // Determine confidence color
        let confidenceColor = '#10b981'; // Green
        if (t.confidence < 70) confidenceColor = '#f59e0b'; // Orange
        if (t.confidence < 50) confidenceColor = '#ef4444'; // Red
        
        const displayName = t.name || `ID ${t.trackId}`;
        
        return `
        <div class="card-item">
            <span class="track-badge">ID ${t.trackId}</span>
            <div style="flex: 1;">
                <strong>${displayName}</strong>
                ${t.confidence > 0 ? `
                    <div style="font-size: 11px; color: ${confidenceColor}; margin-top: 2px;">
                        ${t.confidence.toFixed(0)}% confidence
                    </div>
                ` : ''}
            </div>
            ${t.url ? `
                <a href="${t.url}" target="_blank" style="color: #60a5fa; text-decoration: none;" title="View on Scryfall">
                    🔗
                </a>
            ` : ''}
        </div>
    `}).join('');
}

// Run immediately on script load
init();