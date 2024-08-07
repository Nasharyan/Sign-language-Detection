document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const videoContainer = document.getElementById('video-container');
    const loading = document.getElementById('loading');
    const videoFeed = document.getElementById('video-feed');

    startBtn.addEventListener('click', () => {
        loading.style.display = 'flex';
        videoContainer.style.display = 'block';
        videoFeed.onload = () => {
            loading.style.display = 'none';
        };
    });

    stopBtn.addEventListener('click', () => {
        fetch('/shutdown', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                alert(data.message);
                videoContainer.style.display = 'none';
            });
    });
});