document.addEventListener('DOMContentLoaded', () => {
    // Theme Switcher Logic
    const themeToggleBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const htmlElement = document.documentElement;

    const savedTheme = localStorage.getItem('lumina-theme') || 'dark';
    applyTheme(savedTheme);

    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = htmlElement.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        applyTheme(newTheme);
        localStorage.setItem('lumina-theme', newTheme);
    });

    function applyTheme(theme) {
        htmlElement.setAttribute('data-theme', theme);
        themeIcon.className = 'fa-solid fa-circle-half-stroke';
    }

    // Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const sampleBtn = document.getElementById('sample-btn');
    
    const comparisonView = document.getElementById('comparison-view');
    const beforeImg = document.getElementById('before-img');
    const afterImg = document.getElementById('after-img');
    const afterViewWrapper = document.getElementById('after-view-wrapper');
    const sliderDivider = document.getElementById('slider-divider');
    
    const sharpenSlider = document.getElementById('sharpen-slider');
    const sharpenVal = document.getElementById('sharpen-val');
    const processBtn = document.getElementById('process-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const btnTxt = processBtn.querySelector('.btn-txt');
    const resetBtn = document.getElementById('reset-btn');
    const downloadBtn = document.getElementById('download-btn');
    
    const psnrVal = document.getElementById('psnr-val');
    const ssimVal = document.getElementById('ssim-val');
    const latencyVal = document.getElementById('latency-val');
    
    const projectFilename = document.getElementById('project-filename');
    const projectSpecs = document.getElementById('project-specs');

    let selectedFile = null;

    // 1. Sharpening Slider
    sharpenSlider.addEventListener('input', (e) => {
        sharpenVal.textContent = `${e.target.value}×`;
    });

    // 2. Dropzone & File Input Handlers
    dropzone.addEventListener('click', (e) => {
        if (e.target !== sampleBtn && !sampleBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Preset Sample Image Loader
    sampleBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
            const resp = await fetch('/sample-image');
            if (!resp.ok) throw new Error('Sample image fetch failed');
            const blob = await resp.blob();
            const file = new File([blob], 'valid_0001.png', { type: 'image/png' });
            handleFileSelect(file);
        } catch (err) {
            alert('Could not load preset sample image.');
            console.error(err);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file (JPG, PNG, WEBP).');
            return;
        }
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            beforeImg.src = e.target.result;
            afterImg.src = e.target.result; // Initial baseline placeholder

            dropzone.classList.add('hidden');
            comparisonView.classList.remove('hidden');
            processBtn.disabled = false;
            
            projectFilename.textContent = file.name;
            projectSpecs.textContent = `${file.type.split('/')[1].toUpperCase()} • Ready for 4x Upscale`;

            setSplitPosition(50);
        };
        reader.readAsDataURL(file);
    }

    // Reset Workspace
    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        beforeImg.src = '';
        afterImg.src = '';
        
        comparisonView.classList.add('hidden');
        dropzone.classList.remove('hidden');
        
        processBtn.disabled = true;
        downloadBtn.classList.add('disabled');
        downloadBtn.href = '#';

        psnrVal.textContent = '-- dB';
        ssimVal.textContent = '--';
        latencyVal.textContent = '-- ms';

        projectFilename.textContent = 'No Image Loaded';
        projectSpecs.textContent = 'PNG / JPG • 4x ESRGAN Engine';
    });

    // 3. Upscale API Execution
    processBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        processBtn.disabled = true;
        btnSpinner.classList.remove('hidden');
        btnTxt.style.opacity = '0.5';

        const formData = new FormData();
        formData.append('file', selectedFile);

        const startTime = performance.now();

        try {
            const response = await fetch('/super-resolution', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Inference failed');
            }

            const data = await response.json();
            const clientLatency = Math.round(performance.now() - startTime);

            displayResults(data, clientLatency);

        } catch (error) {
            alert(`Error processing image: ${error.message}`);
            console.error(error);
        } finally {
            processBtn.disabled = false;
            btnSpinner.classList.add('hidden');
            btnTxt.style.opacity = '1.0';
        }
    });

    function displayResults(data, clientLatency) {
        psnrVal.textContent = `+${data.metrics.psnr_vs_bicubic} dB`;
        ssimVal.textContent = data.metrics.ssim_vs_bicubic;
        latencyVal.textContent = `${data.latency_ms || clientLatency} ms`;

        afterImg.src = data.enhanced_image_base64;
        downloadBtn.href = data.enhanced_image_base64;
        downloadBtn.classList.remove('disabled');

        projectSpecs.textContent = `128×128 ➔ 512×512 • PSNR +${data.metrics.psnr_vs_bicubic}dB`;

        setSplitPosition(50);
    }

    // 4. Interactive BEFORE / AFTER Split Dragging
    let isDragging = false;

    sliderDivider.addEventListener('mousedown', () => { isDragging = true; });
    window.addEventListener('mouseup', () => { isDragging = false; });

    comparisonView.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        updateSliderPos(e.clientX);
    });

    sliderDivider.addEventListener('touchstart', () => { isDragging = true; });
    window.addEventListener('touchend', () => { isDragging = false; });
    comparisonView.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        updateSliderPos(e.touches[0].clientX);
    });

    function updateSliderPos(clientX) {
        const rect = comparisonView.getBoundingClientRect();
        let offsetX = clientX - rect.left;
        let percentage = (offsetX / rect.width) * 100;

        if (percentage < 5) percentage = 5;
        if (percentage > 95) percentage = 95;

        setSplitPosition(percentage);
    }

    function setSplitPosition(percentage) {
        sliderDivider.style.left = `${percentage}%`;
        afterViewWrapper.style.width = `${100 - percentage}%`;
    }
});
