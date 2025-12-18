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
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    currentTracks.forEach(track => {
        const x = track.bbox.x * canvas.width;
        const y = track.bbox.y * canvas.height;
        const w = track.bbox.w * canvas.width;
        const h = track.bbox.h * canvas.height;

        // Bounding Box (Matching your Python draw_tracks style)
        ctx.strokeStyle = '#00FF00'; // Green
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        // Centroid Dot
        ctx.fillStyle = '#FF0000'; // Red
        ctx.beginPath();
        ctx.arc(track.centroid.x * canvas.width, track.centroid.y * canvas.height, 3, 0, Math.PI * 2);
        ctx.fill();

        // Label
        ctx.fillStyle = '#00FF00';
        ctx.font = '16px Arial';
        ctx.fillText(`ID ${track.trackId}`, x, y - 5);
    });

    requestAnimationFrame(drawLoop);
}

// 4. Processing Loop (Sending frames to server)
async function logicLoop() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        // OPTIMIZATION: Downscale the image even more! 
        // 480x270 is plenty for DeepSORT to see card colors/patterns
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 480; 
        tempCanvas.height = 270;
        
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 480, 270);
        
        const data = tempCanvas.toDataURL('image/jpeg', 0.5); // Lower quality = faster transfer
        ws.send(JSON.stringify({ type: 'frame', data }));
    }
    // DeepSORT isn't 60FPS. Set this to 150ms-200ms for a smoother experience
    setTimeout(logicLoop, 150); 
}

function updateUIList() {
    const list = document.getElementById('cardsList');
    list.innerHTML = currentTracks.map(t => `
        <div class="card-item">
            <span class="track-badge">ID ${t.trackId}</span>
            <strong>Active</strong>
        </div>
    `).join('');
}

// Run immediately on script load
init();