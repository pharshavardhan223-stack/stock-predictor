document.addEventListener("DOMContentLoaded", function () {

    const recommendations = document.querySelectorAll(".recommendation");

    recommendations.forEach(function (element) {

        const text = element.innerText.trim();

        if (text === "BUY") {
            element.style.color = "green";
        }
        else if (text === "SELL") {
            element.style.color = "red";
        }
        else {
            element.style.color = "orange";
        }
    });

});
