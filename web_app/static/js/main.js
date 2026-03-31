/**
 * Music Recommender - Frontend JavaScript
 * Handles search, filtering, and YouTube player integration
 */

// =============================================================================
// State Management
// =============================================================================

const state = {
    currentMode: 'search', // 'search', 'image', 'mood', 'context'
    isLoading: false,
    results: [],
    selectedImage: null, // For image upload
    recentSearches: JSON.parse(localStorage.getItem('recentSearches')) || []
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    // Inputs
    searchInput: () => document.getElementById('searchInput'),
    artistInput: () => document.getElementById('artistInput'),
    searchBtn: () => document.getElementById('searchBtn'),
    autocompleteDropdown: () => document.getElementById('autocompleteDropdown'),

    // Results
    loading: () => document.getElementById('loading'),
    resultsSection: () => document.getElementById('resultsSection'),
    resultsGrid: () => document.getElementById('resultsGrid'),
    resultsTitle: () => document.getElementById('resultsTitle'),
    resultsCount: () => document.getElementById('resultsCount'),

    // Modal
    youtubeModal: () => document.getElementById('youtubeModal'),
    youtubePlayer: () => document.getElementById('youtubePlayer'),
    modalSongTitle: () => document.getElementById('modalSongTitle'),
    modalArtistName: () => document.getElementById('modalArtistName'),

    // Navigation
    navLinks: () => document.querySelectorAll('.nav-link'),

    // Image Upload
    imageInput: () => document.getElementById('imageInput'),
    imageDropZone: () => document.getElementById('imageDropZone'),
    dropZoneContent: () => document.getElementById('dropZoneContent'),
    imagePreview: () => document.getElementById('imagePreview'),
    imageActions: () => document.getElementById('imageActions'),
    imageMode: () => document.getElementById('image-mode'),
    detectedMoodContainer: () => document.getElementById('detectedMoodContainer'),
    detectedIcon: () => document.getElementById('detectedIcon'),
    detectedLabel: () => document.getElementById('detectedLabel'),
    detectedType: () => document.getElementById('detectedType'),
    confidenceFill: () => document.getElementById('confidenceFill'),
    confidenceText: () => document.getElementById('confidenceText')
};

// =============================================================================
// Mode Switching
// =============================================================================

function switchMode(mode) {
    state.currentMode = mode;

    // Update nav links
    elements.navLinks().forEach(link => {
        link.classList.toggle('active', link.dataset.mode === mode);
    });

    // Show/hide appropriate sections
    const modes = ['search', 'image', 'mood', 'context'];
    modes.forEach(m => {
        const section = document.getElementById(`${m}-mode`);
        if (section) {
            section.classList.toggle('hidden', m !== mode);
        }
    });

    // Clear results when switching modes
    hideResults();

    // Clear image preview when switching modes
    if (mode !== 'image') {
        clearImage();
    }

    // Hide detected mood container when switching modes
    if (elements.detectedMoodContainer()) {
        elements.detectedMoodContainer().classList.add('hidden');
    }
}

// =============================================================================
// Search, Autocomplete & History
// =============================================================================

let autocompleteTimeout;

function updateRecentSearches(query, artist = '') {
    if (!query) return;
    
    const entry = { song: query, artist: artist };
    
    // Remove if already exists
    state.recentSearches = state.recentSearches.filter(
        item => !(item.song.toLowerCase() === query.toLowerCase() && (item.artist || '').toLowerCase() === artist.toLowerCase())
    );
    
    // Add to beginning
    state.recentSearches.unshift(entry);
    
    if (state.recentSearches.length > 10) {
        state.recentSearches.pop();
    }
    
    localStorage.setItem('recentSearches', JSON.stringify(state.recentSearches));
}

function showRecentSearches() {
    const dropdown = elements.autocompleteDropdown();
    
    if (state.recentSearches.length === 0) {
        dropdown.classList.add('hidden');
        return;
    }
    
    let html = '<div class="autocomplete-header">Recent Searches</div>';
    html += state.recentSearches.map(item => `
        <div class="autocomplete-item" onclick="selectAutocomplete('${escapeHtml(item.song)}', '${escapeHtml(item.artist || '')}')">
            <span class="autocomplete-icon">🕒</span>
            <div class="autocomplete-text">
                <div class="autocomplete-song">${escapeHtml(item.song)}</div>
                ${item.artist ? `<div class="autocomplete-artist">${escapeHtml(item.artist)}</div>` : ''}
            </div>
        </div>
    `).join('');
    
    dropdown.innerHTML = html;
    dropdown.classList.remove('hidden');
}

async function handleAutocompleteInput() {
    const query = elements.searchInput().value.trim();
    const dropdown = elements.autocompleteDropdown();
    
    clearTimeout(autocompleteTimeout);
    
    if (!query) {
        showRecentSearches();
        return;
    }
    
    autocompleteTimeout = setTimeout(async () => {
        try {
            // Get selected model
            const searchModelInput = document.querySelector('input[name="searchModel"]:checked');
            const selectedModel = searchModelInput ? searchModelInput.value : 'model1';
            
            const url = new URL('/api/autocomplete', window.location.origin);
            url.searchParams.set('q', query);
            url.searchParams.set('model', selectedModel);
            
            const response = await fetch(url);
            if (!response.ok) throw new Error('Failed to fetch suggestions');
            
            const data = await response.json();
            
            if (data.length === 0) {
                dropdown.classList.add('hidden');
                return;
            }
            
            let html = '<div class="autocomplete-header">Suggestions</div>';
            html += data.map(item => `
                <div class="autocomplete-item" onclick="selectAutocomplete('${escapeHtml(item.song)}', '${escapeHtml(item.artist || '')}')">
                    <span class="autocomplete-icon">🔍</span>
                    <div class="autocomplete-text">
                        <div class="autocomplete-song">${escapeHtml(item.song)}</div>
                        ${item.artist ? `<div class="autocomplete-artist">${escapeHtml(item.artist)}</div>` : ''}
                    </div>
                </div>
            `).join('');
            
            dropdown.innerHTML = html;
            dropdown.classList.remove('hidden');
            
        } catch (error) {
            console.error('Autocomplete error:', error);
            dropdown.classList.add('hidden');
        }
    }, 300);
}

function selectAutocomplete(song, artist) {
    elements.searchInput().value = song;
    elements.artistInput().value = artist;
    elements.autocompleteDropdown().classList.add('hidden');
    handleSearch();
}

async function handleSearch() {
    const query = elements.searchInput().value.trim();
    const artist = elements.artistInput().value.trim();

    // Get selected model
    const searchModelInput = document.querySelector('input[name="searchModel"]:checked');
    const selectedModel = searchModelInput ? searchModelInput.value : 'model1';

    if (!query) {
        showToast('Please enter a song name', 'warning');
        elements.searchInput().focus();
        return;
    }

    showLoading();

    try {
        const url = new URL('/api/search', window.location.origin);
        url.searchParams.set('q', query);
        if (artist) {
            url.searchParams.set('artist', artist);
        }
        url.searchParams.set('model', selectedModel);

        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Search failed');
        }

        const modelName = selectedModel === 'model2' ? 'Model 2 (Audio-Only)' : 'Model 1 (Hybrid)';
        displayResults(data.results, `Similar to "${query}" [${modelName}]`, data.count);
        updateRecentSearches(query, artist);
        elements.autocompleteDropdown().classList.add('hidden');

    } catch (error) {
        console.error('Search error:', error);
        showToast(error.message, 'error');
        hideResults();
    } finally {
        hideLoading();
    }
}

async function searchByMood(mood) {
    showLoading();

    try {
        const response = await fetch(`/api/mood/${mood}?limit=10`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }

        displayResults(data, `${mood} Mood`, data.length);

    } catch (error) {
        console.error('Mood search error:', error);
        showToast(error.message, 'error');
        hideResults();
    } finally {
        hideLoading();
    }
}

async function searchByContext(context) {
    showLoading();

    try {
        const response = await fetch(`/api/context/${context}?limit=10`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }

        displayResults(data, `${context} Vibes`, data.length);

    } catch (error) {
        console.error('Context search error:', error);
        showToast(error.message, 'error');
        hideResults();
    } finally {
        hideLoading();
    }
}


// =============================================================================
// Image Upload Functions
// =============================================================================

function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        previewImage(file);
    }
}

function previewImage(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
    if (!allowedTypes.includes(file.type)) {
        showToast('Please upload a JPEG, PNG, WebP, or GIF image', 'error');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showToast('Image too large. Maximum size is 10MB', 'error');
        return;
    }

    // Store the file
    state.selectedImage = file;

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = elements.imagePreview();
        const dropZoneContent = elements.dropZoneContent();
        const actions = elements.imageActions();

        preview.src = e.target.result;
        preview.classList.remove('hidden');
        dropZoneContent.classList.add('hidden');
        actions.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    state.selectedImage = null;

    const preview = elements.imagePreview();
    const dropZoneContent = elements.dropZoneContent();
    const actions = elements.imageActions();
    const input = elements.imageInput();
    const detectedContainer = elements.detectedMoodContainer();

    if (preview) {
        preview.src = '';
        preview.classList.add('hidden');
    }
    if (dropZoneContent) dropZoneContent.classList.remove('hidden');
    if (actions) actions.classList.add('hidden');
    if (input) input.value = '';
    if (detectedContainer) detectedContainer.classList.add('hidden');
}

async function handleImageUpload() {
    if (!state.selectedImage) {
        showToast('Please select an image first', 'warning');
        return;
    }

    showLoading();

    // Hide detected mood container from previous analysis
    if (elements.detectedMoodContainer()) {
        elements.detectedMoodContainer().classList.add('hidden');
    }

    try {
        // Create FormData and append image
        const formData = new FormData();
        formData.append('file', state.selectedImage);

        // Get selected model
        const selectedModel = document.querySelector('input[name="imageModel"]:checked').value;
        const endpoint = selectedModel === 'model2' ? '/api/recommend/image-v2' : '/api/recommend-from-image';

        // Send to backend
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to analyze image');
        }

        if (!data.success) {
            showToast(data.message || 'Failed to analyze image', 'error');
            hideResults();
            return;
        }

        // Display detected mood/context
        displayDetectedMood(data);

        // Display recommendations
        if (data.recommendations && data.recommendations.length > 0) {
            const modelName = selectedModel === 'model2' ? 'Model 2 (Audio)' : 'Model 1 (Hybrid)';
            displayResults(
                data.recommendations,
                `${data.detected_label} ${data.detected_type === 'mood' ? 'Mood' : 'Activity'} Songs [${modelName}]`,
                data.recommendations.length
            );
        } else {
            showToast('No recommendations found for this mood/context', 'warning');
            hideResults();
        }

    } catch (error) {
        console.error('Image upload error:', error);
        showToast(error.message, 'error');
        hideResults();
    } finally {
        hideLoading();
    }
}

function displayDetectedMood(data) {
    const container = elements.detectedMoodContainer();
    const icon = elements.detectedIcon();
    const label = elements.detectedLabel();
    const type = elements.detectedType();
    const confidenceFill = elements.confidenceFill();
    const confidenceText = elements.confidenceText();

    if (!container) return;

    // Set emoji based on detected label
    const moodEmojis = {
        'Happy': '😊',
        'Sad': '😢',
        'Energetic': '⚡',
        'Calm': '😌',
        'Angry': '😠',
        'Party': '🎉',
        'Workout': '💪',
        'Study': '📚',
        'Relax': '🌸',
        'Driving': '🚗'
    };

    icon.textContent = moodEmojis[data.detected_label] || '🎵';
    label.textContent = data.detected_label;
    type.textContent = data.detected_type === 'mood' ? 'Mood detected' : 'Activity detected';

    const confidencePercent = Math.round(data.confidence * 100);
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceText.textContent = `${confidencePercent}% confidence`;

    container.classList.remove('hidden');
}

function setupImageUploadListeners() {
    const imageInput = elements.imageInput();
    const dropZone = elements.imageDropZone();

    if (!imageInput || !dropZone) return;

    // File input change
    imageInput.addEventListener('change', handleImageSelect);

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');

        const file = e.dataTransfer.files[0];
        if (file) {
            previewImage(file);
        }
    });
}

function displayResults(results, title, count) {
    state.results = results;

    if (!results || results.length === 0) {
        elements.resultsGrid().innerHTML = `
            <div class="no-results" style="grid-column: 1/-1; text-align: center; padding: 60px 20px; color: var(--text-secondary);">
                <p style="font-size: 48px; margin-bottom: 16px;">?</p>
                <p style="font-size: 18px;">No songs found</p>
                <p style="font-size: 14px; margin-top: 8px;">Try a different search term or filter</p>
            </div>
        `;
        elements.resultsTitle().textContent = title;
        elements.resultsCount().textContent = '0 songs';
        elements.resultsSection().classList.remove('hidden');
        return;
    }

    elements.resultsTitle().textContent = title;
    elements.resultsCount().textContent = `${count} songs`;

    elements.resultsGrid().innerHTML = results.map((song, index) => createSongCard(song, index)).join('');

    elements.resultsSection().classList.remove('hidden');

    // Animate cards
    const cards = elements.resultsGrid().querySelectorAll('.song-card');
    cards.forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
            card.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, i * 50);
    });

    // Enrich cards with Spotify album art (non-blocking, runs in background)
    enrichSongCards(results);
}

/**
 * Fetch Spotify metadata for each song in results and inject album art + links.
 * Runs asynchronously so cards appear immediately, art loads progressively.
 */
async function enrichSongCards(results) {
    const enrichPromises = results.map(async (song, index) => {
        try {
            const url = new URL('/api/spotify-enrich', window.location.origin);
            url.searchParams.set('q', song.song);
            if (song.artist) url.searchParams.set('artist', song.artist);

            const res = await fetch(url);
            if (!res.ok) return;
            const data = await res.json();
            if (!data.found) return;

            // Find the card by data-index
            const card = elements.resultsGrid().querySelector(`[data-song-index="${index}"]`);
            if (!card) return;

            // Inject album art
            if (data.album_art) {
                const placeholder = card.querySelector('.album-art-placeholder');
                if (placeholder) {
                    placeholder.innerHTML = `<img src="${data.album_art}" alt="${escapeHtml(song.song)} album art" class="album-art-img" loading="lazy">`;
                    placeholder.classList.add('loaded');
                }
            }

            // Inject Spotify link
            if (data.spotify_url) {
                const spotifyBtn = card.querySelector('.spotify-link-btn');
                if (spotifyBtn) {
                    spotifyBtn.href = data.spotify_url;
                    spotifyBtn.style.display = 'flex';
                }
            }

            // Add explicit badge if needed
            if (data.explicit) {
                const tags = card.querySelector('.song-tags');
                if (tags && !tags.querySelector('.tag-explicit')) {
                    tags.insertAdjacentHTML('beforeend', '<span class="tag tag-explicit">E</span>');
                }
            }

            // Inject duration and popularity
            const infoDiv = card.querySelector('.song-info');
            if (infoDiv) {
                if (data.duration_ms) {
                    const durSpan = infoDiv.querySelector('.song-duration');
                    if (durSpan) durSpan.textContent = ` • ${formatDuration(data.duration_ms)}`;
                }
                if (data.popularity) {
                    const popSpan = infoDiv.querySelector('.popularity-badge');
                    if (popSpan) {
                        popSpan.innerHTML = `🔥 ${data.popularity}`;
                        popSpan.style.fontSize = '12px';
                        popSpan.style.color = 'var(--text-muted)';
                        popSpan.classList.remove('hidden');
                    }
                }
            }

            // Inject 30s preview audio
            if (data.preview_url) {
                const previewContainer = card.querySelector('.preview-container');
                if (previewContainer) {
                    previewContainer.innerHTML = `
                        <div class="audio-preview" onclick="event.stopPropagation()">
                            <audio controls preload="none">
                                <source src="${data.preview_url}" type="audio/mpeg">
                            </audio>
                        </div>
                    `;
                    previewContainer.classList.remove('hidden');
                }
            }
        } catch (e) {
            // Silently fail — enrichment is optional
        }
    });

    // Run all enrichments in parallel (but don't wait)
    Promise.allSettled(enrichPromises);
}

function formatDuration(ms) {
    if (!ms) return '';
    const minutes = Math.floor(ms / 60000);
    const seconds = ((ms % 60000) / 1000).toFixed(0);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
}

function createSongCard(song, index) {
    const similarity = song.similarity !== null && song.similarity !== undefined
        ? `<div class="similarity-score">
               Similarity: ${(song.similarity * 100).toFixed(1)}%
               <div class="similarity-bar">
                   <div class="similarity-fill" style="width: ${song.similarity * 100}%"></div>
               </div>
           </div>`
        : '';

    return `
        <div class="song-card" data-song-index="${index}" onclick="playSong('${escapeHtml(song.song)}', '${escapeHtml(song.artist)}')">
            <div class="album-art-placeholder">
                <span class="album-art-note">🎵</span>
            </div>
            <div class="song-body">
                <div class="song-header">
                    <div class="song-info">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <h3 class="song-name" title="${escapeHtml(song.song)}">${escapeHtml(song.song)}</h3>
                            <span class="popularity-badge hidden"></span>
                        </div>
                        <p class="song-artist" title="${escapeHtml(song.artist)}">
                            ${escapeHtml(song.artist)}<span class="song-duration"></span>
                        </p>
                    </div>
                    <div class="song-actions">
                        <a class="spotify-link-btn" href="#" target="_blank" rel="noopener noreferrer"
                            title="Open on Spotify" style="display:none"
                            onclick="event.stopPropagation()">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                            </svg>
                        </a>
                        <button class="play-btn" title="Play on YouTube"
                            onclick="event.stopPropagation(); playSong('${escapeHtml(song.song)}', '${escapeHtml(song.artist)}')">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M8 5v14l11-7z"/>
                            </svg>
                        </button>
                    </div>
                </div>
                
                <div class="preview-container hidden"></div>

                <div class="song-tags">
                    <span class="tag">${escapeHtml(song.genre)}</span>
                    <span class="tag emotion">${escapeHtml(song.emotion)}</span>
                </div>
                ${similarity}
            </div>
        </div>
    `;
}

// =============================================================================
// YouTube Player
// =============================================================================

async function playSong(song, artist) {
    // Show modal with loading state
    elements.modalSongTitle().textContent = song;
    elements.modalArtistName().textContent = artist;
    elements.youtubePlayer().innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; height: 100%; background: #000;">
            <div class="spinner"></div>
        </div>
    `;
    elements.youtubeModal().classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    try {
        const url = new URL('/api/youtube', window.location.origin);
        url.searchParams.set('song', song);
        url.searchParams.set('artist', artist);

        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Could not find video');
        }

        // Check if this is fallback mode (no video_id means it's a search URL)
        if (!data.video_id || data.video_id === null) {
            // Fallback mode: Open YouTube search in new tab
            // Using anchor element instead of window.open() to avoid pop-up blockers
            const link = document.createElement('a');
            link.href = data.embed_url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';

            // Add to DOM temporarily (required for some browsers)
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            closeModal(); // Close the modal after opening link

            // Show informative message
            const message = data.message || 'Opening YouTube search in new tab...';
            showToast(message, 'info');
            return;
        }

        // Normal mode: Embed video iframe
        elements.youtubePlayer().innerHTML = `
            <iframe 
                src="${data.embed_url}?autoplay=1" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
        `;

    } catch (error) {
        console.error('YouTube error:', error);
        elements.youtubePlayer().innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; background: #000; color: white; text-align: center; padding: 20px;">
                <p style="font-size: 48px; margin-bottom: 16px;">😔</p>
                <p style="font-size: 16px;">${error.message}</p>
                <p style="font-size: 14px; margin-top: 8px; color: #888;">Try searching on YouTube directly</p>
            </div>
        `;
    }
}

function closeModal() {
    elements.youtubeModal().classList.add('hidden');
    elements.youtubePlayer().innerHTML = '';
    document.body.style.overflow = '';
}

// =============================================================================
// UI Helpers
// =============================================================================

function showLoading() {
    state.isLoading = true;
    elements.loading().classList.remove('hidden');
    elements.resultsSection().classList.add('hidden');
}

function hideLoading() {
    state.isLoading = false;
    elements.loading().classList.add('hidden');
}

function hideResults() {
    elements.resultsSection().classList.add('hidden');
}

function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%) translateY(100px);
        background: ${type === 'error' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#8b5cf6'};
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        z-index: 9999;
        transition: transform 0.3s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
        toast.style.transform = 'translateX(-50%) translateY(0)';
    });

    // Remove after delay
    setTimeout(() => {
        toast.style.transform = 'translateX(-50%) translateY(100px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// =============================================================================
// Event Listeners
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Navigation links
    elements.navLinks().forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchMode(link.dataset.mode);
        });
    });

    // Setup image upload listeners
    setupImageUploadListeners();

    // Search on Enter key
    elements.searchInput().addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });

    // Autocomplete focus and input
    elements.searchInput().addEventListener('input', handleAutocompleteInput);
    elements.searchInput().addEventListener('focus', () => {
        if (!elements.searchInput().value.trim()) {
            showRecentSearches();
        } else {
            handleAutocompleteInput();
        }
    });

    // Re-trigger autocomplete when model changes
    document.querySelectorAll('input[name="searchModel"]').forEach(radio => {
        radio.addEventListener('change', () => {
            if (elements.searchInput().value.trim()) {
                handleAutocompleteInput();
            }
        });
    });

    // Close autocomplete when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('#search-mode')) {
            elements.autocompleteDropdown().classList.add('hidden');
        }
    });

    elements.artistInput().addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });


    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !elements.youtubeModal().classList.contains('hidden')) {
            closeModal();
        }
    });
});
