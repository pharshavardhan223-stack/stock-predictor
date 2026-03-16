document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");

    if (!form) return;

    form.addEventListener("submit", function (e) {

        const checkboxes = document.querySelectorAll("input[name='columns']:checked");

        if (checkboxes.length === 0) {
            e.preventDefault();
            alert("Please select at least one column.");
        }

    });

});
