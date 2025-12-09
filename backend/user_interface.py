# Gradio UI for Model Results Display
import gradio as gr
import os
import pandas as pd
from pathlib import Path
import tempfile
import shutil


# Use the copy_plot_to_temp function to allow Gradio to avoid Gradio Invalid Path Exception
# Gradio needs to first copy the file to a temporary folder instead of directly accessing the png file from the main project directory
def copy_plot_to_temp(plot_path):
    if not plot_path or not Path(plot_path).exists():
        return None

    # Create a temp file
    temp_dir = tempfile.gettempdir()
    temp_filename = Path(plot_path).name
    temp_path = os.path.join(temp_dir, temp_filename)

    # Copy file to temp directory
    shutil.copy2(plot_path, temp_path)
    return temp_path


# Function used to access plots from a selected file
def get_ticker_plots(ticker):
    backend_dir = Path(__file__).parent  # Directory where this script is located
    project_dir = backend_dir.parent  # Project directory

    # Try different possible locations
    possible_paths = [
        project_dir / "Graphs" / ticker,
        project_dir / "graphs" / ticker,  # lowercase version
        Path("Graphs") / ticker,
        Path("graphs") / ticker,
        project_dir.parent / "Graphs" / ticker,
        project_dir.parent / "graphs" / ticker,
    ]

    graphs_dir = None
    for path in possible_paths:
        if path.exists():
            graphs_dir = path
            break

    if not graphs_dir:
        return None, f"No graphs directory found for ticker: {ticker}"

    # List of expected file names
    plot_files = {
        "ensemble_residual": "Ensemble_residual_graph.png",
        "f1_performance": "F1_performance_comparison.png",
        "hit_rate_performance": "Hit_Rate_performance_comparison.png",
        "ic_performance": "IC_performance_comparison.png",
        "lstm_residual": "LSTM_residual_graph.png",
        "rmse_performance": "RMSE_performance_comparison.png",
        "xgb_residual": "XGB_residual_graph.png"
    }

    # Check which files exist and copy them to temp directory
    available_plots = {}
    for plot_name, filename in plot_files.items():
        file_path = graphs_dir / filename
        if file_path.exists():
            # Copy to temp directory for Gradio access
            temp_path = copy_plot_to_temp(str(file_path))
            if temp_path:
                available_plots[plot_name] = temp_path

    # Handle the casr for when no file plots are found at all
    if not available_plots:
        return None, f"No plot files found in {graphs_dir}"

    return available_plots, None


# Main function used to pass files to Gradio UI
def display_ticker_results(ticker):
    plots, error = get_ticker_plots(ticker)

    if error:
        return [
            None,  # Residual plot 1
            None,  # Residual plot 2
            None,  # Residual plot 3
            None,  # Metrics plot 1
            None,  # Metrics plot 2
            None,  # Metrics plot 3
            None,  # Metrics plot 4
            error  # Error message
        ]

    # Return image paths organized by category
    return [
        plots.get("xgb_residual", None),  # Tab 1: Residual plots - XGBoost
        plots.get("lstm_residual", None),  # Tab 1: Residual plots - LSTM
        plots.get("ensemble_residual", None),  # Tab 1: Residual plots - Ensemble
        plots.get("rmse_performance", None),  # Tab 2: Metrics plots - RMSE
        plots.get("ic_performance", None),  # Tab 2: Metrics plots - IC
        plots.get("f1_performance", None),  # Tab 2: Metrics plots - F1
        plots.get("hit_rate_performance", None),  # Tab 2: Metrics plots - Hit Rate
        f"✅ Successfully loaded {len(plots)} plots for {ticker}"  # Status message
    ]

# Used by UI to display drop-down menu for ticker select
def get_available_tickers():
    backend_dir = Path(__file__).parent
    project_dir = backend_dir.parent

    # Try different possible locations
    possible_paths = [
        project_dir / "Graphs",
        project_dir / "graphs",  # lowercase version
        Path("Graphs"),
        Path("graphs"),
        project_dir.parent / "Graphs",
        project_dir.parent / "graphs",
    ]

    graphs_dir = None
    for path in possible_paths:
        if path.exists():
            graphs_dir = path
            break

    if not graphs_dir:
        print(f"Warning: No Graphs directory found. Tried: {possible_paths}")
        return []

    # Get all directories in the Graphs folder (assumed to be ticker names)
    tickers = [d.name for d in graphs_dir.iterdir() if d.is_dir()]
    return sorted(tickers)


def load_ticker_data(ticker):
    available_tickers = get_available_tickers()

    if ticker not in available_tickers:
        ticker_list = ", ".join(available_tickers) if available_tickers else "No tickers found"
        return f"❌ Ticker '{ticker}' not found. Available tickers: {ticker_list}", None

    try:
        # Display the plots
        results = display_ticker_results(ticker)

        status = f"✅ Data loaded successfully for {ticker}\n"
        status += f"- Found {sum(1 for r in results[:7] if r is not None)} plot(s)\n"

        return status, *results

    except Exception as e:
        return f"❌ Error loading data: {str(e)}", None, None, None, None, None, None, None, None





#==================================================================================================
# Define Gradio interface
demo = gr.Blocks(title="Stock Prediction Models Dashboard")

with demo:
    gr.Markdown("# 📈 Stock Treadliner")
    gr.Markdown("CS 4210 Machine Learning : Semester Project")
    gr.Markdown("*Insert Group Info Here*")

    # Get available tickers
    available_tickers = get_available_tickers()
    default_ticker = available_tickers[0] if available_tickers else "AAPL"

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Data Configuration")

            # Create dropdown with available tickers
            if available_tickers:
                ticker_input = gr.Dropdown(
                    label="Stock Ticker",
                    choices=available_tickers,
                    value=default_ticker,
                    info="Select a ticker to view its performance graphs"
                )
            else:
                ticker_input = gr.Textbox(
                    label="Stock Ticker",
                    value="AAPL",
                    placeholder="Enter stock ticker (e.g., AAPL, AMZN, GOOGL, MSFT, TSLA)",
                    info="Note: No tickers found in Graphs directory"
                )

            load_btn = gr.Button("📥 Load and Display Plots", variant="primary")
            data_status = gr.Textbox(label="Data Status", lines=3, interactive=False)

            gr.Markdown("---")
            gr.Markdown("### ℹ️ Available Tickers")

            if available_tickers:
                ticker_list = "\n".join([f"- {ticker}" for ticker in available_tickers])
                gr.Markdown(f"**Model was trained on the following tickers:**\n{ticker_list}")
            else:
                gr.Markdown("**No tickers found.**\nPlease ensure the Graphs directory exists with ticker subfolders.")

            gr.Markdown("---")
            gr.Markdown("### 📊 Plot Information")

            plot_info = gr.Markdown("""
            **Plot Categories:**

            **📉 Residual Plots** (Tab 1):
            1. **XGB Residual Graph**: XGBoost model residuals
            2. **LSTM Residual Graph**: LSTM model residuals  
            3. **Ensemble Residual Graph**: Ensemble model residuals

            **📈 Performance Metrics** (Tab 2):
            1. **RMSE Performance**: Root Mean Square Error comparison
            2. **IC Performance**: Information Coefficient comparison
            3. **F1 Performance**: F1-score comparison
            4. **Hit Rate Performance**: Hit rate comparison
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 📈 Model Performance Graphs")

            # Split content into tabs for better readability
            with gr.Tabs():
                # First tab lists model plots
                with gr.TabItem("📉 Residual Plots"):
                    gr.Markdown("### Model Residual Analysis")
                    gr.Markdown("Visualizing prediction errors across different models")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**XGBoost Residuals**")
                            xgb_plot = gr.Image(label="", type="filepath", height=300)
                        with gr.Column():
                            gr.Markdown("**LSTM Residuals**")
                            lstm_plot = gr.Image(label="", type="filepath", height=300)

                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("**Ensemble Model Residuals**")
                            ensemble_plot = gr.Image(label="", type="filepath", height=300)
                        with gr.Column(scale=1):
                            residual_info = gr.Markdown("""
                            **About Residuals:**
                            - Residuals are calculated by subtracting the Actual Value by the Predicted Value
                            - Good Model Performance is reflected by Low Residual
                            - Patterns in residuals indicate model bias
                            """)

                # Second tab lists Performance Metrics
                with gr.TabItem("📈 Performance Metrics"):
                    gr.Markdown("### Model Performance Comparison")
                    gr.Markdown("Comparing different evaluation metrics across models")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**RMSE Performance**")
                            rmse_plot = gr.Image(label="", type="filepath", height=280)
                        with gr.Column():
                            gr.Markdown("**Information Coefficient**")
                            ic_plot = gr.Image(label="", type="filepath", height=280)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**F1 Score Performance**")
                            f1_plot = gr.Image(label="", type="filepath", height=280)
                        with gr.Column():
                            gr.Markdown("**Hit Rate Performance**")
                            hit_rate_plot = gr.Image(label="", type="filepath", height=280)

                    metrics_info = gr.Markdown("""
                    **Metrics Explanation:**
                    - **RMSE**: Lower indicates better prediction accuracy
                    - **IC**: Values closer to 1 indicate a model with better predictive capability
                    - **F1 Score**: Higher indicates a good balance between precision and recall
                    - **Hit Rate**: Higher indicates model is more likely correct than not
                    """)

            # Status message area
            gr.Markdown("---")
            plot_status = gr.Textbox(label="Plot Loading Status", lines=2, interactive=False)

    # Event handler
    load_btn.click(
        fn=load_ticker_data,
        inputs=[ticker_input],
        outputs=[
            data_status,
            xgb_plot,  # Tab 1: Residual plots
            lstm_plot,  # Tab 1: Residual plots
            ensemble_plot,  # Tab 1: Residual plots
            rmse_plot,  # Tab 2: Metrics plots
            ic_plot,  # Tab 2: Metrics plots
            f1_plot,  # Tab 2: Metrics plots
            hit_rate_plot,  # Tab 2: Metrics plots
            plot_status  # Status message
        ]
    )

# CSS Styling
css = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}
h1 {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 20px;
}
h3 {
    color: #34495e;
    margin-top: 15px;
}
.tab-nav {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 10px;
}
.tab-button {
    font-weight: bold !important;
}
.plot-title {
    font-weight: bold;
    margin-bottom: 8px;
    color: #2c3e50;
    text-align: center;
}
.plot-container {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 15px;
    background: white;
}
.info-box {
    background: #f0f7ff;
    border-left: 4px solid #4a90e2;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
}
"""

demo.css = css

gr.Markdown("""
### 🎯 How to use:
1. **Select a stock ticker** from the dropdown
2. **Click 'Load and Display Plots'** to load all graphs
3. **Switch between tabs** to view:
   - **Residual Plots**: Model error analysis
   - **Performance Metrics**: Model comparison charts

### 📁 File Structure:
The application looks for plot images in: `project_directory/Graphs/{TICKER}/`

**Expected plot files for each ticker:**
- `XGB_residual_graph.png`
- `LSTM_residual_graph.png`
- `Ensemble_residual_graph.png`
- `RMSE_performance_comparison.png`
- `IC_performance_comparison.png`
- `F1_performance_comparison.png`
- `Hit_Rate_performance_comparison.png`
""")

# Show the demo
print("✅ Gradio UI created successfully!")
print(f"Available tickers: {get_available_tickers()}")
print("\nTo launch the dashboard, run:")
print("demo.launch()")
print("\nOr with custom settings:")
print("demo.launch(server_name='0.0.0.0', server_port=7860, share=True)")

demo.launch(share = True)