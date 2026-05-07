const fileInput = document.getElementById("pdf_file");
const fileName = document.getElementById("file-name");

if (fileInput && fileName) {
    fileInput.addEventListener("change", () => {
        const selected = fileInput.files && fileInput.files.length > 0 ? fileInput.files[0].name : "No file selected";
        fileName.textContent = selected;
    });
}
