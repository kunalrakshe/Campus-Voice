function setExample(text) {
    const textarea = document.getElementById('feedback_text');
    if (textarea) {
        textarea.value = text;
        // Small highlight effect to show it changed
        textarea.style.backgroundColor = '#EEF2FF';
        setTimeout(() => {
            textarea.style.backgroundColor = '';
        }, 300);
    }
}
