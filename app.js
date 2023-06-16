function analyzeRepositories() {
    var username = document.getElementById("username").value;
    var data = { 'username': username };

    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        // Handle the response and update the UI with the analysis results
        displayAnalysisResult(result);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function displayAnalysisResult(result) {
    var resultDiv = document.getElementById("result");
    resultDiv.innerHTML = `
        <h2>Most Complex Repository:</h2>
        <p>Name: ${result.most_complex_repository.name}</p>
        <p>URL: <a href="${result.most_complex_repository.url}" target="_blank">${result.most_complex_repository.url}</a></p>
        <p>Description: ${result.most_complex_repository.description}</p>
        <p>Complexity Score: ${result.most_complex_repository.complexity_score}</p>
        <h2>GPT Analysis:</h2>
        ${generateGPTAnalysisHTML(result.gpt_analysis)}
    `;
}

function generateGPTAnalysisHTML(gptAnalysis) {
    // Generate the HTML for the GPT analysis
    // Customize based on the structure of the analysis response
    // Return the HTML string
}
