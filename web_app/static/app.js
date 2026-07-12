// Frontend controller logic for IMDB Sentiment Analyzer

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const reviewInput = document.getElementById("review-input");
    const charCounter = document.getElementById("char-counter");
    const clearInputBtn = document.getElementById("clear-input-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const templateChips = document.querySelectorAll(".template-chip");
    
    const resultsEmpty = document.getElementById("results-empty");
    const resultsContent = document.getElementById("results-content");
    const modelStatus = document.getElementById("model-status");
    
    // Results DOM elements
    const gaugePath = document.getElementById("gauge-path");
    const gaugePct = document.getElementById("gauge-pct");
    const gaugeLabel = document.getElementById("gauge-label");
    const sentimentBadge = document.getElementById("sentiment-badge");
    const confidenceVal = document.getElementById("confidence-val");
    const verdictDesc = document.getElementById("verdict-desc");
    
    // Step Elements
    const stepRaw = document.getElementById("step-raw");
    const stepCleaned = document.getElementById("step-cleaned");
    const stepPunc = document.getElementById("step-punc");
    const stepStopwordsText = document.getElementById("step-stopwords-text");
    const removedStopwordsBox = document.getElementById("removed-stopwords-box");
    const removedStopwordsChips = document.getElementById("removed-stopwords-chips");
    const stepStemmed = document.getElementById("step-stemmed");
    const stepCards = document.querySelectorAll(".step-card");
    
    // History DOM elements
    const historyEmpty = document.getElementById("history-empty");
    const historyList = document.getElementById("history-list");
    const clearHistoryBtn = document.getElementById("clear-history-btn");
    
    // Model stats DOM elements
    const statAccuracy = document.getElementById("stat-accuracy");
    const statVocab = document.getElementById("stat-vocab");
    const statHidden = document.getElementById("stat-hidden");
    const statModelType = document.getElementById("stat-model-type");
    
    // Toast Element
    const toast = document.getElementById("toast");

    // Theme Toggle DOM Elements
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const themeIcon = document.getElementById("theme-icon");

    // History state
    let history = JSON.parse(localStorage.getItem("sentiment_history") || "[]");

    // 1. Initialize Application
    initTheme();
    checkModelStatus();
    renderHistory();
    
    function initTheme() {
        const savedTheme = localStorage.getItem("theme_pref");
        if (savedTheme === "light") {
            document.body.classList.add("light-theme");
            themeIcon.className = "fa-solid fa-sun";
        } else {
            document.body.classList.remove("light-theme");
            themeIcon.className = "fa-solid fa-moon";
        }
    }
    
    // 2. Event Listeners
    
    // Theme toggle click handler
    themeToggleBtn.addEventListener("click", () => {
        const isLight = document.body.classList.toggle("light-theme");
        if (isLight) {
            themeIcon.className = "fa-solid fa-sun";
            localStorage.setItem("theme_pref", "light");
        } else {
            themeIcon.className = "fa-solid fa-moon";
            localStorage.setItem("theme_pref", "dark");
        }
    });
    
    // Textarea input char counter
    reviewInput.addEventListener("input", () => {
        const len = reviewInput.value.length;
        charCounter.textContent = `${len} / 2000 chars`;
    });

    // Clear input button
    clearInputBtn.addEventListener("click", () => {
        reviewInput.value = "";
        charCounter.textContent = "0 / 2000 chars";
        reviewInput.focus();
    });

    // Template click handler
    templateChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const text = chip.getAttribute("data-text");
            reviewInput.value = text;
            charCounter.textContent = `${text.length} / 2000 chars`;
            analyzeSentiment();
        });
    });

    // Analyze button click
    analyzeBtn.addEventListener("click", analyzeSentiment);

    // Collapsible Step Cards
    stepCards.forEach(card => {
        const header = card.querySelector(".step-header");
        header.addEventListener("click", () => {
            // Close other steps
            stepCards.forEach(c => {
                if (c !== card) c.classList.remove("active");
            });
            // Toggle current step
            card.classList.toggle("active");
        });
    });

    // Clear History Button
    clearHistoryBtn.addEventListener("click", () => {
        history = [];
        localStorage.setItem("sentiment_history", JSON.stringify(history));
        renderHistory();
    });

    // 3. Functions
    
    // Check API stats and connection on load
    async function checkModelStatus() {
        try {
            const response = await fetch("/stats");
            if (!response.ok) throw new Error("Server error");
            const data = await response.json();
            
            if (data.status === "trained") {
                modelStatus.className = "status-badge connected";
                modelStatus.querySelector(".status-text").textContent = "Model Connected";
                
                // Update stats fields
                statAccuracy.textContent = `${data.accuracy}%`;
                statVocab.textContent = `${data.vocabulary_size.toLocaleString()} features`;
                statHidden.textContent = `${data.hidden_size} nodes`;
                statModelType.textContent = data.model_type;
            } else {
                modelStatus.className = "status-badge";
                modelStatus.querySelector(".status-text").textContent = "Untrained";
                showToast("Model is not trained yet! Please run training script first.", "warning");
            }
        } catch (error) {
            modelStatus.className = "status-badge";
            modelStatus.querySelector(".status-text").textContent = "Offline";
            showToast("Cannot connect to backend server. Make sure FastAPI is running.", "error");
        }
    }

    // Call prediction endpoint
    async function analyzeSentiment() {
        const reviewText = reviewInput.value.trim();
        if (!reviewText) {
            showToast("Please enter some movie review text first.", "warning");
            return;
        }

        // Set Loading State
        analyzeBtn.disabled = true;
        const originalBtnHTML = analyzeBtn.innerHTML;
        analyzeBtn.innerHTML = `<span>Analyzing...</span> <i class="fa-solid fa-circle-notch spinner"></i>`;

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ review: reviewText })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Prediction failed");
            }

            const data = await response.json();
            displayResult(data);
            addToHistory(reviewText, data.sentiment, data.confidence);
            
            // Auto scroll to results on mobile devices
            if (window.innerWidth <= 900) {
                document.getElementById("results-panel").scrollIntoView({ behavior: "smooth" });
            }

        } catch (error) {
            showToast(error.message, "error");
        } finally {
            // Restore Button State
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = originalBtnHTML;
        }
    }

    // Update results interface
    function displayResult(data) {
        resultsEmpty.classList.add("hidden");
        resultsContent.classList.remove("hidden");

        const sentiment = data.sentiment;
        const confidence = data.confidence;
        const steps = data.steps;

        // Set Badge and Classes
        sentimentBadge.textContent = sentiment;
        sentimentBadge.className = `badge ${sentiment.toLowerCase()}`;
        
        confidenceVal.textContent = `${(confidence * 100).toFixed(1)}%`;
        confidenceVal.style.color = sentiment === "Positive" ? "var(--positive)" : "var(--negative)";

        // Set Verdict description
        if (sentiment === "Positive") {
            verdictDesc.textContent = `The model predicted positive sentiment with ${(confidence * 100).toFixed(1)}% confidence, indicating the reviewer enjoyed the movie.`;
        } else {
            verdictDesc.textContent = `The model predicted negative sentiment with ${(confidence * 100).toFixed(1)}% confidence, indicating a critical or negative review.`;
        }

        // Animate Radial Gauge Path
        // dashoffset = circumference - (percent * circumference)
        // positive maps to 100%, negative maps to 0%
        // Let's use the positive probability (from 0 to 1) for the gauge position
        const positiveProb = sentiment === "Positive" ? confidence : (1 - confidence);
        const circumference = 314; // 2 * PI * r = 2 * 3.14159 * 50
        const strokeDashoffset = circumference - (positiveProb * circumference);
        
        gaugePath.style.strokeDashoffset = strokeDashoffset;
        gaugePath.style.stroke = sentiment === "Positive" ? "var(--positive)" : "var(--negative)";
        
        gaugePct.textContent = `${Math.round(positiveProb * 100)}%`;
        gaugeLabel.textContent = "Positive Pct";

        // Fill NLP Preprocessing steps inspector
        stepRaw.textContent = steps.raw;
        stepCleaned.textContent = steps.lowercased !== steps.urls_removed 
            ? `[URL Cleaned] ${steps.html_removed}` 
            : steps.html_removed;
        stepPunc.textContent = steps.punctuations_removed;
        stepStopwordsText.textContent = steps.stopwords_removed;
        stepStemmed.textContent = steps.stemmed;

        // Populate Stopwords Chips
        if (steps.removed_stopwords_list && steps.removed_stopwords_list.length > 0) {
            removedStopwordsChips.innerHTML = "";
            steps.removed_stopwords_list.forEach(word => {
                const chip = document.createElement("span");
                chip.className = "stopword-badge";
                chip.textContent = word;
                removedStopwordsChips.appendChild(chip);
            });
            removedStopwordsBox.classList.remove("hidden");
        } else {
            removedStopwordsBox.classList.add("hidden");
        }
        
        // Auto open step 4 (Stopwords) or 5 (Stemmed) to make it interactive
        stepCards.forEach(c => c.classList.remove("active"));
        document.querySelector('[data-step="4"]').classList.add("active");
    }

    // Add query to local history
    function addToHistory(text, sentiment, confidence) {
        const item = {
            id: Date.now(),
            text: text,
            sentiment: sentiment,
            confidence: confidence,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        
        // Limit history to 10 items
        history.unshift(item);
        if (history.length > 10) history.pop();
        
        localStorage.setItem("sentiment_history", JSON.stringify(history));
        renderHistory();
    }

    // Render local history
    function renderHistory() {
        if (history.length === 0) {
            historyEmpty.classList.remove("hidden");
            historyList.classList.add("hidden");
            return;
        }

        historyEmpty.classList.add("hidden");
        historyList.classList.remove("hidden");
        historyList.innerHTML = "";

        history.forEach(item => {
            const li = document.createElement("li");
            li.className = "history-item";
            
            const badgeClass = item.sentiment.toLowerCase();
            
            li.innerHTML = `
                <span class="hist-lbl ${badgeClass}">${item.sentiment[0]}</span>
                <span class="hist-text" title="Click to reload this text">${escapeHtml(item.text)}</span>
                <div class="hist-actions">
                    <span class="hist-conf">${(item.confidence * 100).toFixed(0)}%</span>
                    <button class="delete-hist-btn" data-id="${item.id}" title="Remove entry">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </div>
            `;
            
            // Click on text reloads the query
            li.querySelector(".hist-text").addEventListener("click", () => {
                reviewInput.value = item.text;
                charCounter.textContent = `${item.text.length} / 2000 chars`;
                analyzeSentiment();
            });

            // Delete single item handler
            li.querySelector(".delete-hist-btn").addEventListener("click", (e) => {
                e.stopPropagation();
                deleteHistoryItem(item.id);
            });

            historyList.appendChild(li);
        });
    }

    function deleteHistoryItem(id) {
        history = history.filter(item => item.id !== id);
        localStorage.setItem("sentiment_history", JSON.stringify(history));
        renderHistory();
    }

    // Toast notifications utility
    function showToast(message, type = "error") {
        toast.className = "toast";
        if (type === "warning") {
            toast.style.borderLeftColor = "#f59e0b";
            toast.querySelector("i").className = "fa-solid fa-triangle-exclamation";
            toast.querySelector("i").style.color = "#f59e0b";
        } else {
            toast.style.borderLeftColor = "var(--negative)";
            toast.querySelector("i").className = "fa-solid fa-circle-exclamation";
            toast.querySelector("i").style.color = "var(--negative)";
        }
        
        toast.querySelector(".toast-msg").textContent = message;
        toast.classList.remove("hidden");

        // Clear after 4 seconds
        if (window.toastTimeout) clearTimeout(window.toastTimeout);
        window.toastTimeout = setTimeout(() => {
            toast.classList.add("hidden");
        }, 4000);
    }

    // Helper to safely render strings in HTML
    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
