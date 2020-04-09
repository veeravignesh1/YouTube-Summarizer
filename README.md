<h1 align="center"> YouTube-Summarizer</h1>

## Problem Statement

We spend a considerable amount of time on YouTube. Which is a primary source of knowledge because it contains lecture videos from primer institutes. We watch a hour long video and try to grab the important ideas from it. What if we can let a system do it. Understand the important ideas from the video and summarize it in a way which will allow you to look back and revisit the key ideas without watching the video all over again

## Objective

Our project uses Advanced NLP techniques to extract the summary of the video. Its always difficult to run these codes for a common user. What if we have a GUI which allows us to just paste in the link of the YouTube video and the application does all the hard lifting for us and saves the summary in the text format in the desired location.

## Process

- Extract the closed caption in English from the user given YouTube link
- Clean the Data by removing all the time stamps and make it a single document
- Run the NLP text summarization algorithm on the given document
- Develop a GUI to link the process
- Location to save the summary in

## Working
[GUI](gui.gif)
[Result](result.gif)

## Future Work

- Extend the summarization to Coursera and other education websites
- On Paste of the entire playlist extract all, process all and save each of them in separate files
