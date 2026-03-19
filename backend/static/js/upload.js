document.addEventListener("DOMContentLoaded", function () {

    const fileInput = document.querySelector("input[type='file']");

    if (!fileInput) return;

    fileInput.addEventListener("change", function () {

        const file = this.files[0];

        if (!file) return;

        const allowedExtensions = ["csv"];
        const extension = file.name.split(".").pop().toLowerCase();

        if (!allowedExtensions.includes(extension)) {
            alert("Only CSV files are allowed.");
            this.value = "";
            return;
        }

        const maxSize = 5 * 1024 * 1024; // 5MB

        if (file.size > maxSize) {
            alert("File size must be less than 5MB.");
            this.value = "";
        }
    });

});
