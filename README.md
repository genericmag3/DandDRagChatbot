# Local TTRPG Campaign RAG Chatbot

## Summary
This project is an early implementation of a local RAG chatbot that will take dated notes as input, semantically chunk the notes and convert them into a vector database, and then pass relevant notes as context for queries about the campaign to the chatbot.
The chatbot will provide references to relevant note chunks at the end of each response. If there are no relevant notes found, the LLM is bypassed and a canned response is outputted. Custom animations have also been added for some extra flavor.

## Features
- User can vectorize own notes in .txt or .docx format. Ideally, these notes are date-delimited for better parsing.
- User can specify party member names and from which party member's perspective the notes are written from. These options are passed to the default user prompt.
- References are interactable buttons that display the reference material retrived by semantic similarity search. Buttons are labeled with reference note dates pulled from vector database entry metadata.
- Interfaces with local Ollama installation, allowing for dynamic LLM and temperature selection.

## Demo
[Demo Video](https://drive.google.com/file/d/1uaDG5qKW1TUAhX0aV6PVxxXFYU1kDjre/view?usp=sharing)


