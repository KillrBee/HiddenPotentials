import json
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedTokenizerFast
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

hidden_states = []
AllOutputs = []  # the list of outputs
attentions = []
all_tokens = []
selected_prompts = []
logits = []
#E:\Downloads\_old\text-generation-webui\models\TheBloke_airoboros-33B-gpt4-1-4-SuperHOT-8K-GPTQ
model_name = "/mnt/e/Downloads/text-generation-webui/text-generation-webui/models/ehartford_WizardLM-33B-V1.0-Uncensored"
#model_name = "/mnt/e/Downloads/_Nous-Hms-L2-13b/"
#model_name = "E:\Downloads\_Nous-Hms-L2-13b"

def load_all_prompts():
    # Ask the user whether they want to enter a prompt or load prompts from a domain
    man_prompt = input("Please enter your prompt: ")
    prompts = [man_prompt]
    # choice = input(
        # 'Do you want to enter a single prompt or load prompts from a domain? Enter "(S)ingle" or "(D)omain": '
    # )
    # if choice.lower() == "s":
        # # The user wants to enter a single prompt
        # man_prompt = input("Please enter your prompt: ")
        # prompts = [man_prompt]
    # elif choice.lower() == "d":
        # # Load all prompts
        # with open("HiddenStatePrompts.json", "r") as file:
            # all_prompts = json.load(file)

        # # Get the list of domain dictionaries
        # domains = all_prompts["domains"]

        # # Print out the domain names
        # for i, domain in enumerate(domains):
            # print(f"{i + 1}. {domain['name']}")
        # # The user wants to load prompts from a domain
        # domain_index = (
            # int(input("Please enter the number of the domain you want to select: ")) - 1
        # )

        # # Check if the provided index is valid
        # if domain_index < 0 or domain_index >= len(domains):
            # raise ValueError(
                # f"Invalid index {domain_index + 1}. Please enter a number between 1 and {len(domains)}."
            # )

        # # Get the prompts for the selected domain
        # prompts = domains[domain_index]["prompts"]
    # else:
        # raise ValueError('Invalid choice. Please enter "S" or "D".')
    selected_prompts.append(prompts)
    return selected_prompts


def load_model_and_encode(prompts):
    # Clear the GPU memory cache
    torch.cuda.empty_cache()

    #  Load model and tokenizer
    print(f"Loading Tokenizer:")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)  
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Setting Configuration Profile")
    config = AutoConfig.from_pretrained(
        model_name,
        init_device="meta",
        trust_remote_code=True,
        output_hidden_states=True,
        output_attentions=True,
        #torch_dtype="bfloat16",
        do_sample=True,
        max_new_tokens=128
    )

    # Load the model
    print(f"Loading the Model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, trust_remote_code=True, load_in_4bit=True) #, torch_dtype=torch.float16)
   
    print(f"Finished Loading the Model")
    # Encode prompts and get hidden states
    print(f"Encoding Prompts and getting Internal Model Data")
    for prompt in selected_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        prompt_tokens = []
        for i, token in enumerate(tokens):
            prompt_tokens.append(token)
            all_tokens.append(prompt_tokens)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            return_dict=True) 
            #do_sample=True, 
            #max_new_tokens=512)
        AllOutputs.append(outputs.logits)
        hidden_states.append(outputs.hidden_states)
        attentions.append(outputs.attentions)

    return hidden_states, attentions, all_tokens

def plot_attention_heatmaps(attentions, all_tokens):
    for i, attention in enumerate(attentions):
        # Attention weights should have a shape similar to [num_layers, num_heads, seq_len, seq_len]
        # Here we're averaging across the heads for simplicity
        attention = attention.mean(dim=1).detach().numpy()
        
        # Create a heatmap for each layer
        for j, layer_attention in enumerate(attention):
            fig = go.Figure(
                data=go.Heatmap(
                    z=layer_attention,
                    x=all_tokens[i],
                    y=all_tokens[i],
                    colorscale="Jet",
                    hovertemplate=
                    "<b>Token</b>: %{x}<br>"
                    + "<b>Token</b>: %{y}<br>"
                    + "<b>Weight</b>: %{z}<extra></extra>",
                )
            )
            fig.update_layout(title=f"Attention Weights for Prompt {i + 1}, Layer {j + 1}")
            fig.update_xaxes(title_text="Tokens")
            fig.update_yaxes(title_text="Tokens", autorange="reversed")
            fig.show()


def plot_depth_graph(hidden_states, all_tokens):
    figgraph = go.Figure()
    total_layers = len(hidden_states[0])  # Assuming all hidden states have the same number of layers
    print(f"Starting Surface Plot Generation")
    for i in range(total_layers):
            for j, hidden_state in enumerate(hidden_states):
                data = hidden_state[i][0].float().detach().numpy()

                # # Debug Code
                # print(f"Debugging data types")
                # print(f"------------------------------")
                # print(type(hidden_states))
                # print(type(hidden_states[0]))
                # print(type(hidden_states[0][0]))
                # print(hidden_states[0][0].dtype)
                # print(f"------------------------------")
                # print(f"Unique data types")
                # print(f"------------------------------")
                # unique_data_types = set()
                # for state in hidden_states:
                    # for layer in state:
                        # unique_data_types.update(np.unique(layer.detach().numpy().dtype))
                # print(unique_data_types)
                # print(f"------------------------------")
                # print(f"Checking for NaN")
                # print(f"------------------------------")
                # contains_nan_or_inf = any(np.isnan(hidden_state).any() or np.isinf(hidden_state).any() for hidden_state in hidden_states)
                # print(f"Contains NaN or infinite values: {contains_nan_or_inf}")
                # print(f"------------------------------")
                # print(f"Checking for problem elements")
                # print(f"------------------------------")
                # for i, state in enumerate(hidden_states):
                    # for j, layer in enumerate(state):
                        # data = layer.detach().numpy()
                        # if data.dtype not in [np.float32, np.float64]:
                            # print(f"Unexpected datatype at hidden_state[{i}][{j}]: {data.dtype}")
                        # if np.isnan(data).any():
                            # print(f"NaN value detected at hidden_state[{i}][{j}]")
                        # if np.isinf(data).any():
                            # print(f"Infinite value detected at hidden_state[{i}][{j}]")
                
                # print(f"------------END DEBUG----------------------")

                
                x_values = [f'Hidden_State {k}' for k in range(data.shape[1])]
                y_values = [f'{all_tokens[j][l]}' for l in range(data.shape[0])]

                # Filter out data with the start token '<s>'
                indices_to_remove = [index for index, token in enumerate(y_values) if token == '<s>']
                for index in reversed(indices_to_remove):
                    y_values.pop(index)
                    data = np.delete(data, index, axis=0)
                # Plot the data    
                surface = go.Surface(x=x_values, y=y_values, z=data, cmin=np.min(data), cmax=np.max(data), showscale=True, colorscale="Viridis")
                figgraph.add_trace(surface)

    for trace in figgraph.data:
        trace.visible = False

    if len(figgraph.data) > 0:
        figgraph.data[0].visible = True

    updatemenu = []
    buttons = []
    for i in range(total_layers):
        visibility = [i == j for j in range(total_layers)]
        button = dict(
            label = f"Layer {i + 1}",
            method = 'update',
            args = [{'visible': visibility},
                    {'title': f"Layer {i + 1}"}]
        )
        buttons.append(button)
    updatemenu.append(buttons)

    figgraph.update_layout(title_text="Hidden States Activation Plot", updatemenus=[{"buttons": updatemenu[0],
                                         "direction": "down",
                                         "active": 0,
                                         "showactive": True,
                                         "x": 0.1,
                                         "y": 1.1}],
                                         scene=dict(
                                            yaxis=dict(tickvals=list(range(len(y_values))), ticktext=y_values))
                                        )
    figgraph.update_traces(contours_x_highlightwidth=3,selector=dict(type='surface'))
    
    figgraph.show()


def plot_scatter_graph(hidden_states, all_tokens):
    figgraph = go.Figure()

    total_layers = len(hidden_states[0])  # Assuming all hidden states have the same number of layers

    # Extract token labels (assuming consistent tokens across all hidden states)
    token_labels = [f'{all_tokens[0][l]}' for l in range(hidden_states[0][0][0].shape[0])]
    indices_to_remove = [index for index, token in enumerate(token_labels) if token == '<s>']
    for index in reversed(indices_to_remove):
        token_labels.pop(index)

    # Add 3D Scatter Plot traces
    for i in range(total_layers):
        for j, hidden_state in enumerate(hidden_states):
            data = hidden_state[i][0].detach().numpy()

            x_values = np.array([k for k in range(data.shape[1]) for _ in range(data.shape[0])])
            y_values = np.array([l for _ in range(data.shape[1]) for l in range(data.shape[0])])
            z_values = data.flatten()

            scatter_3d = go.Scatter3d(x=x_values, y=y_values, z=z_values, mode='markers', marker=dict(size=5, color=y_values, colorscale='Viridis', opacity=0.8), visible=False, name=f"3D Scatter Layer {i+1}")
            figgraph.add_trace(scatter_3d)

    # Add 3D Surface Plot traces
    for i in range(total_layers):
        for j, hidden_state in enumerate(hidden_states):
            data = hidden_state[i][0].detach().numpy()

            surface = go.Surface(z=data, colorscale='Viridis', cmin=np.min(data), cmax=np.max(data), visible=False, name=f"Surface Layer {i+1}")
            figgraph.add_trace(surface)

    # Create buttons to toggle between 3D Scatter and 3D Surface
    buttons = []
    for i in range(total_layers):
        visibility = [False] * (2 * total_layers)  # Initialize all to False
        visibility[i] = True  # 3D Scatter Plot
        button = dict(
            label = f"3D Scatter Layer {i + 1}",
            method = 'update',
            args = [{'visible': visibility},
                    {'title': f"3D Scatter Layer {i + 1}"}]
        )
        buttons.append(button)

        visibility = [False] * (2 * total_layers)  # Initialize all to False
        visibility[i + total_layers] = True  # 3D Surface Plot
        button = dict(
            label = f"Surface Layer {i + 1}",
            method = 'update',
            args = [{'visible': visibility},
                    {'title': f"Surface Layer {i + 1}"}]
        )
        buttons.append(button)

    figgraph.update_layout(updatemenus=[{"buttons": buttons,
                                         "direction": "down",
                                         "active": 0,
                                         "showactive": True,
                                         "x": 0.1,
                                         "y": 1.1,
                                         "type": "buttons"}],
                                        scene=dict(
                                            yaxis=dict(tickvals=list(range(len(token_labels))), ticktext=token_labels)
                                        )
                            )
    figgraph.show()


def plot_acvtivation_heatmap(hidden_states, all_tokens):

    total_layers = len(hidden_states[0])  # Assuming all hidden states have the same number of layers
    num_tokens = hidden_states[0][0][0].shape[0]

    # Initialize an array to store average values for each token in each layer
    avg_values = np.zeros((num_tokens, total_layers))
    print(f"Starting Heatmap Generation")
    # Calculate average values
    for i in range(total_layers):
        for j, hidden_state in enumerate(hidden_states):
            data = hidden_state[i][0].detach().numpy()
            avg_values[:, i] += data[:, 0]
        avg_values[:, i] /= len(hidden_states)

    # Filter out rows associated with '<s>'
    token_labels = [f'{all_tokens[0][l]}' for l in range(num_tokens)]
    indices_to_remove = [index for index, token in enumerate(token_labels) if token == '<s>']
    avg_values = np.delete(avg_values, indices_to_remove, axis=0)
    for index in reversed(indices_to_remove):
        token_labels.pop(index)

    # Calculate mean and standard deviation
    mean_val = np.mean(avg_values)/2
    std_val = np.std(avg_values)/2
    
    # Normalize the colorscale breakpoints
    min_val = np.min(avg_values)
    max_val = np.max(avg_values)
    lower_bound = (mean_val - std_val - min_val) / (max_val - min_val)
    upper_bound = (mean_val + std_val - min_val) / (max_val - min_val)

     # Manually define the Spectral colorscale
    spectral_scale = [
        "#9e0142",
        "#d53e4f",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#e6f598",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
        "#5e4fa2"
    ]

    # Create a custom color scale
    custom_colorscale = [
        [0.0, spectral_scale[0]],
        [lower_bound, "lightslategrey"],
        [upper_bound, "lightsteelblue"],
        [1.0, spectral_scale[-1]]
    ]

    # Create 2D heatmap with custom color scale
    heatmap = go.Heatmap(z=avg_values, x=[f'Layer {k+1}' for k in range(total_layers)], y=token_labels, colorscale=custom_colorscale)

    fig = go.Figure(data=[heatmap])
    fig.update_layout(title="Heatmap of Hidden State Activation")
    fig.show()

def plot_histogram(hidden_states, all_tokens):
    total_layers = len(hidden_states[0])
    num_tokens = hidden_states[0][0][0].shape[0]

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis

    print(f"Plotting Bargraph plot")
    # Initial plot with average values
    avg_values = np.mean(np.array([[hidden_state[i][0].detach().numpy()[:, 0] for i in range(total_layers)] for hidden_state in hidden_states]), axis=0)
    num_tokens, total_layers = avg_values.shape
    xpos, ypos = np.meshgrid(np.arange(total_layers), np.arange(num_tokens), indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.75
    dz = avg_values.flatten()
    
    # Calculate mean and standard deviation
    mean_val = np.mean(avg_values)
    std_val = np.std(avg_values)

    # Normalize the colorscale breakpoints
    min_val = np.min(avg_values)
    max_val = np.max(avg_values)
    lower_bound = (mean_val - std_val - min_val) / (max_val - min_val)
    upper_bound = (mean_val + std_val - min_val) / (max_val - min_val)

    # Manually define the Spectral colorscale
    spectral_scale = [
        "#9e0142",
        "#d53e4f",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#e6f598",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
        "#5e4fa2"
    ]

    # Create a custom color scale
    colors = [(0.0, spectral_scale[0]),
              (lower_bound, "lightslategrey"),
              (upper_bound, "lightsteelblue"),
              (1.0, spectral_scale[-1])]

    # Create the colormap
    cmap_name = "custom_diverging"
    cm_custom = LinearSegmentedColormap.from_list(cmap_name, colors)

    # Normalize the z-values for color mapping
    norm = plt.Normalize(dz.min(), dz.max())
    colors = cm.Spectral(norm(dz))
    
    # Normalize the dz values to [0, 1]
    norm_dz = (dz - dz.min()) / (dz.max() - dz.min())

    # Get the colors for each bar using the custom colormap
    bar_colors = cm_custom(norm_dz)

    # Plot the bars with the custom colors
    bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=bar_colors)

    ax.set_xlabel('Tokens')
    ax.set_ylabel('Layer')
    ax.set_zlabel('Avg. Hidden State Activation')
    
    # ax.set_xticks(np.arange(total_layers))
    ax.set_xticklabels([f'Layer {i+1}' for i in range(total_layers)], rotation=90, ha='right')

    #ax.set_xticks(np.arange(num_tokens))
    ax.set_xticklabels(all_tokens[0])
    plt.show()

load_all_prompts()
print(f"Loading Model: {model_name}")

hidden_states, attentions, all_tokens = load_model_and_encode(selected_prompts)

print(f"Analyzed Tokens: {all_tokens}")

print(f"Starting Plot Generation")

# Create empty go.Figure() object
fig = go.Figure()

# Generate the plots
#plot_scatter_graph(hidden_states, all_tokens)
plot_depth_graph(hidden_states, all_tokens)

# Generate plots that show the complete end to end activations
plot_acvtivation_heatmap(hidden_states, all_tokens)
#plot_histogram(hidden_states, all_tokens)

# TODO: 
#plot_attention_heatmaps(attentions, all_tokens)

print(f"Finished")


