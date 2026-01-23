/**
 * Music Recommender - Frontend JavaScript
 * Handles search, filtering, and YouTube player integration
 */

// =============================================================================
// State Management
// =============================================================================

const state = {
    currentMode: 'search', // 'search', 'mood', 'context'
    isLoading: false,
    results: []
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    // Inputs
    searchInput: () => document.getElementById('searchInput'),
    artistInput: () => document.getElementById('artistInput'),
    searchBtn: () => document.getElementById('searchBtn'),

    // Sections
    searchMode: () => document.getElementById('search-mode'),
    moodMode: () => document.getElementById('mood-mode'),
    contextMode: () => document.getElementById('context-mode'),

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
    navLinks: () => document.querySelectorAll('.nav-link')
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
    const modes = ['search', 'mood', 'context'];
    modes.forEach(m => {
        const section = document.getElementById(`${m}-mode`);
        if (section) {
            section.classList.toggle('hidden', m !== mode);
        }
    });

    // Clear results when switching modes
    hideResults();
}

// =============================================================================
// Search Functions
// =============================================================================

async function handleSearch() {
    const query = elements.searchInput().value.trim();
    const artist = elements.artistInput().value.trim();

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

        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Search failed');
        }

        displayResults(data.results, `Similar to "${query}"`, data.count);

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
// Display Functions
// =============================================================================

function displayResults(results, title, count) {
    state.results = results;

    if (!results || results.length === 0) {
        elements.resultsGrid().innerHTML = `
            <div class="no-results" style="grid-column: 1/-1; text-align: center; padding: 60px 20px; color: var(--text-secondary);">
                <p style="font-size: 48px; margin-bottom: 16px;">🔍</p>
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
        <div class="song-card" onclick="playSong('${escapeHtml(song.song)}', '${escapeHtml(song.artist)}')">
            <div class="song-header">
                <div class="song-info">
                    <h3 class="song-name" title="${escapeHtml(song.song)}">${escapeHtml(song.song)}</h3>
                    <p class="song-artist" title="${escapeHtml(song.artist)}">${escapeHtml(song.artist)}</p>
                </div>
                <button class="play-btn" title="Play on YouTube" onclick="event.stopPropagation(); playSong('${escapeHtml(song.song)}', '${escapeHtml(song.artist)}')">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                </button>
            </div>
            <div class="song-tags">
                <span class="tag">${escapeHtml(song.genre)}</span>
                <span class="tag emotion">${escapeHtml(song.emotion)}</span>
            </div>
            ${similarity}
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

    // Search on Enter key
    elements.searchInput().addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
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
