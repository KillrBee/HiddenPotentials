# HiddenPotentials
Analyze the Hidden Potential of LLMs tp perform on un-trained tasks.

Provides a simple method to examine the activation of Hidden Layers / Hidden Potentials found within an LLM. HiddenStatePrompts.json provide a series of prompts that will be fed into the code for analysis. These can be selected by selecing [D]omain prompts. If you wish to just produce analysis on a single prompt, you can select [S]ingle and type in your prompt directly.

This is meant as a rudamentary script to produce various output visualizations, currently it produces a surface plot of all the hidden layer activations for each processing layer (selectable in the dropdown), a heatmap of the average hidden state activations of each layer for each token, and a 3d-barchart that expresses a sense of how the model gains understanding and context as each layer is processed. This final 3d-barchart had to be created using pyplot as plotly doesn't generate 3d Barcharts, and due to this, it blocks the processing thread execution until the window is closed (and I'm just moving on from the code although I'm sure there's a way to spawn it's own execution thread and not block the code).
