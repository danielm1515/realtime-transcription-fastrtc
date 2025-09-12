// Wait for the DOM to be fully loaded before running any code
document.addEventListener('DOMContentLoaded', function() {
    // Global variables for WebRTC connection
    let peerConnection;      // Stores the WebRTC connection object for audio streaming
    let webrtc_id;           // A unique ID to identify this connection on the server
    let audioContext, analyser, audioSource;  // Audio processing objects for visualization
    let audioLevel = 0;      // Stores the current audio level (volume) from 0-1
    let animationFrame;      // Reference to the animation frame for audio visualization
    let isRecording = false; // Tracks whether we're currently recording or not
    let eventSource;         // Object that receives transcription results from the server
    let llmEventSource;      // Object that receives LLM responses from the server

    // DOM element references
    const startButton = document.getElementById('start-button');    // The button to start/stop recording
    const transcriptDiv = document.getElementById('transcript');    // The container for transcription text
    const pulseButton = document.getElementById('pulse-button');    // The pulse button for recording
    const pulseStatus = document.getElementById('pulse-status');    // Status text below pulse

    // Log debug info at start
    console.log('DOM loaded. startButton:', startButton, 'transcriptDiv:', transcriptDiv);

    // Variables for managing the transcript display
    let currentParagraph = null;    // Reference to the current paragraph being updated
    let lastUpdateTime = Date.now(); // Timestamp of when we last updated the transcript
    let currentLLMParagraph = null;  // Reference to the current LLM response paragraph

    // Show error messages to the user in a toast notification
    function showError(message) {
        console.error('Error:', message);
        const toast = document.getElementById('error-toast');   // Get the toast element
        toast.textContent = message;                           // Set the error message
        toast.style.display = 'block';                         // Make the toast visible

        // Hide toast after 5 seconds
        setTimeout(() => {
            toast.style.display = 'none';                      // Hide the toast
        }, 5000);
    }

    // Handle messages received from the server through WebRTC data channel
    function handleMessage(event) {
        // Parse JSON message
        const eventJson = JSON.parse(event.data);
        // Display errors to the user
        if (eventJson.type === "error") {
            showError(eventJson.message);
        }
        // Log all messages to console for debugging
        console.log('Received message:', event.data);
    }

    // Update button appearance based on connection state
    function updateButtonState() {
        // If connecting, show spinner
        if (peerConnection && (peerConnection.connectionState === 'connecting' || peerConnection.connectionState === 'new')) {
            startButton.innerHTML = `
                <div class="icon-with-spinner">
                    <div class="spinner"></div>
                    <span>Connecting...</span>
                </div>
            `;
            if (pulseStatus) {
                pulseStatus.textContent = 'Connecting...';
                pulseStatus.classList.remove('recording');
            }
            isRecording = false;  // Not recording while connecting
        // If connected, show pulsing recording indicator
        } else if (peerConnection && peerConnection.connectionState === 'connected') {
            startButton.innerHTML = `
                <div class="pulse-container">
                    <div class="pulse-circle"></div>
                    <span>Stop Recording</span>
                </div>
            `;
            if (pulseStatus) {
                pulseStatus.textContent = 'Recording... Click to stop';
                pulseStatus.classList.add('recording');
            }
            isRecording = true;   // Set recording state to true
        // Default state - ready to start
        } else {
            startButton.innerHTML = 'Start Recording';
            if (pulseStatus) {
                pulseStatus.textContent = 'Click to start recording';
                pulseStatus.classList.remove('recording');
            }
            isRecording = false;  // Not recording when not connected
        }
        console.log('Button state updated. isRecording:', isRecording);
    }

    // Set up audio visualization to show when the user is speaking
    function setupAudioVisualization(stream) {
        // Create or resume the audio context
        if (!audioContext) {
            // Create new audio context with browser compatibility handling
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } else {
            // Resume context if it was suspended
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }
        
        // Create audio analyzer for processing audio data
        analyser = audioContext.createAnalyser();
        // Create media source from microphone stream
        audioSource = audioContext.createMediaStreamSource(stream);
        // Connect source to analyzer
        audioSource.connect(analyser);
        // Set FFT size (controls frequency data resolution)
        analyser.fftSize = 64;
        // Create array to store frequency data
        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Function to continuously update audio level visualization
        function updateAudioLevel() {
            // Get audio frequency data
            analyser.getByteFrequencyData(dataArray);
            // Calculate average volume across all frequencies
            const average = Array.from(dataArray).reduce((a, b) => a + b, 0) / dataArray.length;
            // Convert to 0-1 scale
            audioLevel = average / 255;

            // Update pulse and rings based on audio level
            const pulse = document.querySelector('.pulse');
            const pulseRings = document.querySelectorAll('.pulse-ring');
            
            if (pulse) {
                // Simple scaling based on audio level (1.0 to 1.2)
                const scale = 1 + (audioLevel * 0.2);
                pulse.style.setProperty('--audio-level', scale);
                
                // Simple recording state - just add/remove recording class
                if (audioLevel > 0.1) {
                    pulse.classList.add('recording');
                } else {
                    pulse.classList.remove('recording');
                }
                
                // Control color animation speed based on audio level
                if (audioLevel > 0.3) {
                    pulse.style.setProperty('--color-speed', '0.5s'); // Very fast when loud
                } else if (audioLevel > 0.2) {
                    pulse.style.setProperty('--color-speed', '1s');   // Fast when speaking
                } else if (audioLevel > 0.1) {
                    pulse.style.setProperty('--color-speed', '2s');   // Medium when quiet speech
                } else {
                    pulse.style.setProperty('--color-speed', '8s');   // Slow when idle
                }
            }
            
            // Simple ring animation based on audio level
            if (pulseRings) {
                pulseRings.forEach((ring, index) => {
                    if (audioLevel > 0.1) {
                        ring.classList.add('active');
                        // Simple speed variation
                        const speed = Math.max(1.5, 3 - audioLevel * 2);
                        ring.style.animationDuration = `${speed}s`;
                    } else {
                        ring.classList.remove('active');
                        ring.style.animationDuration = '3s';
                    }
                });
            }

            // Continue animation loop
            animationFrame = requestAnimationFrame(updateAudioLevel);
        }
        // Start audio visualization loop
        updateAudioLevel();
    }

    // Set up WebRTC connection for streaming audio to server
    async function setupWebRTC() {
        console.log('Setting up WebRTC connection...');
        
        try {
            // Get WebRTC configuration from global variable
            const config = window.__RTC_CONFIGURATION__;
            console.log('WebRTC configuration:', config);
            
            // Create new peer connection
            peerConnection = new RTCPeerConnection(config);
            console.log('Created peer connection:', peerConnection);

            // Set connection timeout (15 seconds)
            const connectionTimeout = setTimeout(() => {
                if (peerConnection && peerConnection.connectionState !== 'connected') {
                    showError('Connection timeout. Please check your network and try again.');
                    stop(); // Stop connection attempt
                }
            }, 15000);

            // Set warning for slow connection (5 seconds)
            const timeoutId = setTimeout(() => {
                const toast = document.getElementById('error-toast');
                toast.textContent = "Connection is taking longer than usual. Are you on a VPN?";
                toast.className = 'toast warning';
                toast.style.display = 'block';

                // Hide warning after 5 seconds
                setTimeout(() => {
                    toast.style.display = 'none';
                }, 5000);
            }, 5000);

            // Update button to show connecting state
            updateButtonState();

            // Request access to user's microphone
            console.log('Requesting microphone access...');
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: true // Only request audio access
            });
            console.log('Microphone access granted:', stream);

            // Set up audio visualization
            setupAudioVisualization(stream);

            // Add audio tracks to WebRTC connection
            stream.getTracks().forEach(track => {
                peerConnection.addTrack(track, stream);
            });
            console.log('Added audio tracks to connection');

            // Monitor connection state changes
            peerConnection.addEventListener('connectionstatechange', () => {
                // Log state changes
                console.log('connectionstatechange', peerConnection.connectionState);
                
                // Handle successful connection
                if (peerConnection.connectionState === 'connected') {
                    clearTimeout(timeoutId);
                    clearTimeout(connectionTimeout);
                    const toast = document.getElementById('error-toast');
                    toast.style.display = 'none';
                    console.log('Connection established successfully');
                // Handle connection failures
                } else if (peerConnection.connectionState === 'failed' || 
                          peerConnection.connectionState === 'disconnected' || 
                          peerConnection.connectionState === 'closed') {
                    showError('Connection lost. Please try again.');
                    stop();
                }
                // Update button appearance
                updateButtonState();
            });

            // Create data channel for server messages
            const dataChannel = peerConnection.createDataChannel('text');
            dataChannel.onmessage = handleMessage;  // Set message handler
            console.log('Created data channel');

            // Add ICE candidate handler to send candidates as they're discovered
            peerConnection.onicecandidate = ({ candidate }) => {
                if (candidate) {
                    console.log("Sending ICE candidate", candidate);
                    fetch('/webrtc/offer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            candidate: candidate.toJSON(),
                            webrtc_id: webrtc_id,
                            type: "ice-candidate",
                        })
                    });
                }
            };

            // Create connection offer
            console.log('Creating connection offer...');
            const offer = await peerConnection.createOffer();
            // Set local description (our end of connection)
            await peerConnection.setLocalDescription(offer);
            console.log('Local description set');

            // Generate random ID for this connection
            webrtc_id = Math.random().toString(36).substring(7);
            console.log('Generated webrtc_id:', webrtc_id);

            // Send connection offer to server immediately
            console.log('Sending offer to server...');
            const response = await fetch('/webrtc/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: peerConnection.localDescription.sdp,        // Session description
                    type: peerConnection.localDescription.type,      // Offer type
                    webrtc_id: webrtc_id                             // Unique connection ID
                })
            });
            console.log('Server responded to offer');

            // Parse server response
            const serverResponse = await response.json();
            console.log('Server response:', serverResponse);

            // Handle server errors
            if (serverResponse.status === 'failed') {
                showError(serverResponse.meta.error === 'concurrency_limit_reached'
                    ? `Too many connections. Maximum limit is ${serverResponse.meta.limit}`
                    : serverResponse.meta.error);
                stop();
                startButton.textContent = 'Start Recording';
                return;
            }

            // Complete connection with server's description
            console.log('Setting remote description...');
            await peerConnection.setRemoteDescription(serverResponse);
            console.log('Remote description set');

            // Create event source for receiving transcription results
            console.log('Creating event source for transcription...');
            eventSource = new EventSource('/transcript?webrtc_id=' + webrtc_id);
            // Handle event source errors
            eventSource.onerror = (event) => {
                console.error("EventSource error:", event);
                showError("Transcription connection lost. Please try again.");
            };
            // Process transcription results as they arrive
            eventSource.addEventListener("output", (event) => {
                console.log("Received transcript chunk:", event.data);
                // Add text to display
                appendTranscript(event.data);
                
                // Start LLM response stream if transcript is not empty
                if (event.data.trim()) {
                    startLLMResponseStream();
                }
            });
            
            console.log('WebRTC setup complete, waiting for connection...');
        } catch (err) {
            // Handle any setup errors
            console.error('Error setting up WebRTC:', err);
            showError('Failed to establish connection: ' + err.message);
            stop();
            startButton.textContent = 'Start Recording';
        }
    }

    // Start LLM response stream
    function startLLMResponseStream() {
        if (llmEventSource) {
            llmEventSource.close();
        }
        
        console.log('Creating LLM response stream...');
        llmEventSource = new EventSource('/llm-response?webrtc_id=' + webrtc_id);
        
        llmEventSource.onerror = (event) => {
            console.error("LLM EventSource error:", event);
        };
        
        llmEventSource.addEventListener("llm-output", (event) => {
            console.log("Received LLM chunk:", event.data);
            appendLLMResponse(event.data);
        });
        
        llmEventSource.addEventListener("error", (event) => {
            console.error("LLM processing error:", event.data);
            showError("LLM processing error: " + event.data);
        });
    }

    // Add LLM response to display
    function appendLLMResponse(text) {
        if (!text.trim()) return;
        
        // Create new LLM response paragraph if none exists
        if (!currentLLMParagraph) {
            currentLLMParagraph = document.createElement('p');
            currentLLMParagraph.classList.add('llm-response');
            currentLLMParagraph.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
            currentLLMParagraph.style.borderLeft = '3px solid #10b981';
            currentLLMParagraph.style.paddingLeft = '12px';
            currentLLMParagraph.style.fontStyle = 'italic';
            transcriptDiv.appendChild(currentLLMParagraph);
            currentLLMParagraph.textContent = '';
        }
        
        // Append new text to current LLM paragraph
        currentLLMParagraph.textContent += text;
        
        // Auto-scroll to show new content
        requestAnimationFrame(() => {
            transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
        });
    }

    function appendTranscriptSimple(text) {
        const p = document.createElement('p');
        p.textContent = text;
        transcriptDiv.appendChild(p);
        transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
    }

    // Add transcription text to display
    function appendTranscript(text) {
        // Clean up text
        const formattedText = text.trim();
        if (!formattedText) return;
        
        // Reset LLM paragraph for new transcript
        currentLLMParagraph = null;
        
        const now = Date.now();
        const timeSinceLastUpdate = now - lastUpdateTime;
        lastUpdateTime = now;
        
        // Handle transcript display
        if (!currentParagraph) {
            // Create new paragraph
            currentParagraph = document.createElement('p');
            currentParagraph.classList.add('current');
            transcriptDiv.appendChild(currentParagraph);
            currentParagraph.textContent = formattedText;
        } else {
            // Get current text
            const currentText = currentParagraph.textContent;
            
            // Fix spacing issues by normalizing
            let cleanedText = formattedText;
            
            // 1. Check for simple word repetition - last word repeated
            const words = currentText.split(/\s+/);
            const lastWord = words[words.length - 1].replace(/[^\w]/g, '').toLowerCase();
            
            if (lastWord && lastWord.length > 2) {
                // Check if new text starts with the same word
                const regex = new RegExp(`^${lastWord}`, 'i');
                if (regex.test(cleanedText.replace(/[^\w]/g, ''))) {
                    // Remove the first word if it's a duplicate
                    cleanedText = cleanedText.replace(regex, '').trim();
                }
            }
            
            // 2. Add proper spacing
            let finalText = currentText;
            
            // Only add space if current text doesn't end with space or punctuation
            // and new text doesn't start with punctuation
            if (!/[\s.,!?]$/.test(finalText) && !/^[.,!?]/.test(cleanedText) && cleanedText) {
                finalText += ' ';
            }
            
            // 3. Add the cleaned text
            finalText += cleanedText;
            
            // 4. Fix any run-together words by adding spaces after punctuation
            finalText = finalText.replace(/([.,!?])([a-zA-Z])/g, '$1 $2');
            
            // Update the paragraph text
            currentParagraph.textContent = finalText;
        }
        
        // Create new paragraph on sentence end or pause
        if (/[.!?]$/.test(formattedText) || timeSinceLastUpdate > 5000) {
            // End current paragraph
            if (currentParagraph) {
                currentParagraph.classList.remove('current');
            }
            
            // Prepare for next paragraph
            currentParagraph = null;
        }
        
        // Limit number of displayed paragraphs
        const paragraphs = transcriptDiv.getElementsByTagName('p');
        while (paragraphs.length > 10) { // Keep last 10 paragraphs
            transcriptDiv.removeChild(paragraphs[0]);
        }
        
        // Scroll to show newest text
        requestAnimationFrame(() => {
            transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
        });
    }

    // Stop recording and clean up resources
    function stop() {
        console.log('Stopping recording...');
        // Stop audio visualization
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
            animationFrame = null;
        }
        
        // Pause audio processing
        if (audioContext) {
            audioContext.suspend();
        }
        
        // Stop all media tracks
        if (peerConnection) {
            const senders = peerConnection.getSenders();
            if (senders) {
                senders.forEach(sender => {
                    if (sender.track) {
                        sender.track.stop();  // Release microphone
                    }
                });
            }
            
            // Close WebRTC connection
            peerConnection.close();
            peerConnection = null;
        }
        
        // Close transcription connection
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        
        // Close LLM response connection
        if (llmEventSource) {
            llmEventSource.close();
            llmEventSource = null;
        }
        
        // Reset audio level
        audioLevel = 0;
        // Update button display
        updateButtonState();
        
        // Ask about clearing transcript
        if (window.confirm('Clear transcript?')) {
            // Clear all transcript text
            transcriptDiv.innerHTML = '';
            currentParagraph = null;
        } else {
            // Just end current paragraph
            if (currentParagraph) {
                currentParagraph.classList.remove('current');
                currentParagraph = null;
            }
        }
        
        // Reset LLM paragraph
        currentLLMParagraph = null;
        
        // Reset timestamp
        lastUpdateTime = Date.now();
        console.log('Recording stopped');
    }

    // Clean up resources when page is closed
    window.addEventListener('beforeunload', () => {
        stop();  // Stop recording and release resources
    });

    // Handle start/stop button clicks
    function handleRecordingToggle() {
        console.log('Recording toggle clicked. isRecording:', isRecording);
        if (!isRecording) {
            // Start recording if not already recording
            setupWebRTC();
        } else {
            // Stop recording if currently recording
            stop();
        }
    }

    startButton.addEventListener('click', handleRecordingToggle);
    
    // Also add click handler to pulse button
    if (pulseButton) {
        pulseButton.addEventListener('click', handleRecordingToggle);
    }

    // Initialize UI when page loads
    console.log('Initializing UI...');
    // Ensure all UI elements are visible
    const elementsToCheck = [
        transcriptDiv,
        startButton,
        document.getElementById('error-toast')
    ];
    
    // Set appropriate display for each element
    elementsToCheck.forEach(el => {
        if (el) {
            // Set appropriate display style based on element type
            el.style.display = el.tagName.toLowerCase() === 'button' ? 'block' : 
                              (el.id === 'transcript' ? 'block' : 'none');
        }
    });
    
    // Apply CSS variables to ensure theme is working
    document.body.style.backgroundColor = 'var(--background-dark)';
    document.body.style.color = 'var(--text-light)';
    
    // Force button colors for consistency
    startButton.style.backgroundColor = 'rgba(249, 164, 92, 1.0)';
    startButton.style.color = 'black';
    
    console.log('UI initialization complete');
});