document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");

    if (!form) return;

    form.addEventListener("submit", function (e) {

        const username = document.querySelector("input[name='username']").value.trim();
        const password = document.querySelector("input[name='password']").value.trim();

        if (username === "" || password === "") {
            e.preventDefault();
            alert("Please enter both username and password.");
        }
    });

});
