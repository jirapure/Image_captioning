function previewImage(event) {
  const reader = new FileReader();
  reader.onload = function () {
    const imagePreview = document.getElementById("image-preview");
    const imagePreviewContainer = document.getElementById(
      "image-preview-container"
    );
    const imageUpload = document.getElementById("image_upload");
    imagePreview.src = reader.result;
    // imagePreviewContainer.classList.remove("image-preview-container");
    if (
      imageUpload.files.length === 0 ||
      !imageUpload.files[0].type.startsWith("image")
    ) {
      imagePreviewContainer.style.display = "none"; 
      alert("Please upload an image.");
    } else {
      imagePreviewContainer.style.display = "block";
    }
  };
  reader.readAsDataURL(event.target.files[0]);
}

document
  .getElementById("caption-form")
  .addEventListener("submit", function (event) {
    const imageUpload = document.getElementById("image_upload");
    if (
      imageUpload.files.length === 0 ||
      !imageUpload.files[0].type.startsWith("image")
    ) {
      alert("Please upload an image.");
      event.preventDefault();
    }
  });

const copyButtons = document.querySelectorAll(".copy-button");
copyButtons.forEach((button) => {
  button.addEventListener("click", (event) => {
    const caption = button.parentElement.innerText.trim();
    try {
      const textarea = document.createElement("textarea");
      textarea.value = caption;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      button.classList.add("copied");
      setTimeout(() => {
        button.classList.remove("copied");
      }, 1000);
    } catch (err) {
      console.error("Failed to copy caption:", err);
    }
  });
});
