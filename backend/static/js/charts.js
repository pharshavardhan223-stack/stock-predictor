function renderChart(chartData, chartType = "line") {

    let traces = [];

    for (let column in chartData) {

        let trace = {
            x: Array.from({ length: chartData[column].length }, (_, i) => i),
            y: chartData[column],
            name: column,
            type: chartType === "bar" ? "bar" : "scatter",
            mode: chartType === "line" ? "lines" : undefined
        };

        traces.push(trace);
    }

    let layout = {
        title: "Stock Data Visualization",
        xaxis: { title: "Index" },
        yaxis: { title: "Value" }
    };

    Plotly.newPlot("chart", traces, layout);
}
