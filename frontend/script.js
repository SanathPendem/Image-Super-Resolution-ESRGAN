document.addEventListener('DOMContentLoaded', () => {
    // Theme Switcher Logic
    const themeToggleBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');
    const htmlElement = document.documentElement;

    const savedTheme = localStorage.getItem('esrgan-theme') || 'dark';
    applyTheme(savedTheme);

    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = htmlElement.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        applyTheme(newTheme);
        localStorage.setItem('esrgan-theme', newTheme);
    });

    function applyTheme(theme) {
        htmlElement.setAttribute('data-theme', theme);
        themeIcon.className = 'fa-solid fa-circle-half-stroke';
        if (theme === 'light') {
            themeText.textContent = 'Dark Mode';
        } else {
            themeText.textContent = 'Light Mode';
        }
    }

    // Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewContainer = document.getElementById('preview-container');
    const lrPreviewImg = document.getElementById('lr-preview-img');
    const removeBtn = document.getElementById('remove-btn');
    const sampleBtn = document.getElementById('sample-btn');
    
    const sharpenSlider = document.getElementById('sharpen-slider');
    const sharpenVal = document.getElementById('sharpen-val');
    
    const processBtn = document.getElementById('process-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const btnText = processBtn.querySelector('.btn-text');
    
    const placeholderState = document.getElementById('placeholder-state');
    const resultGrid = document.getElementById('result-grid');
    const uploadedImg = document.getElementById('uploaded-img');
    const enhancedImg = document.getElementById('enhanced-img');
    const actionBar = document.getElementById('action-bar');
    const downloadLink = document.getElementById('download-link');
    
    const psnrVal = document.getElementById('psnr-val');
    const ssimVal = document.getElementById('ssim-val');
    const latencyVal = document.getElementById('latency-val');

    let selectedFile = null;

    // 1. Sharpening Slider Input
    sharpenSlider.addEventListener('input', (e) => {
        sharpenVal.textContent = `${e.target.value}×`;
    });

    // 2. Dropzone Click & Drag Event Handlers
    dropzone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-active');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-active');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-active');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFileInput();
    });

    // 3. Preset Sample Loader Button
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
            alert('Please select a valid image file (JPG or PNG).');
            return;
        }
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            // Update left preview in dropzone
            lrPreviewImg.src = e.target.result;
            dropzonePrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            processBtn.disabled = false;

            // Immediately reveal side-by-side 2-card output grid
            uploadedImg.src = e.target.result;
            enhancedImg.src = e.target.result; // Baseline until enhanced button clicked

            placeholderState.classList.add('hidden');
            resultGrid.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    function resetFileInput() {
        selectedFile = null;
        fileInput.value = '';
        lrPreviewImg.src = '';
        uploadedImg.src = '';
        enhancedImg.src = '';
        dropzonePrompt.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        processBtn.disabled = true;

        placeholderState.classList.remove('hidden');
        resultGrid.classList.add('hidden');
        actionBar.classList.add('hidden');

        psnrVal.textContent = '-- dB';
        ssimVal.textContent = '--';
        latencyVal.textContent = '-- ms';
    }

    // 4. Image Enhancement API Request
    processBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        processBtn.disabled = true;
        btnSpinner.classList.remove('hidden');
        btnText.style.opacity = '0.5';

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
            btnText.style.opacity = '1.0';
        }
    });

    function displayResults(data, clientLatency) {
        psnrVal.textContent = `+${data.metrics.psnr_vs_bicubic} dB`;
        ssimVal.textContent = data.metrics.ssim_vs_bicubic;
        latencyVal.textContent = `${data.latency_ms || clientLatency} ms`;

        uploadedImg.src = lrPreviewImg.src;
        enhancedImg.src = data.enhanced_image_base64;
        downloadLink.href = data.enhanced_image_base64;

        placeholderState.classList.add('hidden');
        resultGrid.classList.remove('hidden');
        actionBar.classList.remove('hidden');
    }
});
