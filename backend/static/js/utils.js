function showAlert(message, type = "info") {

    let color;

    if (type === "error") color = "red";
    else if (type === "success") color = "green";
    else color = "blue";

    alert(message);
}


function formatNumber(value, decimals = 2) {
    return parseFloat(value).toFixed(decimals);
}
