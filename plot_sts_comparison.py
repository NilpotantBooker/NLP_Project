import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_json_data(filepath):
    """Loads JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"错误：在 {filepath} 未找到文件") # Console message in Chinese
        return None
    except json.JSONDecodeError as e:
        print(f"错误：无法从 {filepath} 解码JSON。详细信息: {e}") # Console message in Chinese
        return None
    except Exception as e:
        print(f"加载 {filepath} 时发生意外错误: {e}") # Console message in Chinese
        return None

def plot_overall_sts_comparison(encoder_list, sts_tasks_list, scores_data, model_name="Qwen/Qwen2-7B-Instruct", output_image_file="sts_overall_comparison_by_encoder.png"):
    """
    Generates a bar chart with two subplots comparing encoders across multiple STS tasks
    for overall Pearson and Spearman 'all' scores. X-axis is encoder, bars are datasets.

    Args:
        encoder_list (list): List of encoder names (e.g., ['PromptEOL', 'DistillCore']).
        sts_tasks_list (list): List of STS task names (e.g., ['STS12', 'STS13']).
        scores_data (dict): A dictionary containing the scores.
                             Format: {'EncoderName': {'pearson': [scores], 'spearman': [scores]}}
                             The order of scores should match sts_tasks_list.
        model_name (str): The name of the model used for the title.
        output_image_file (str): Filename to save the plot image.
    """
    if not scores_data or not encoder_list or not sts_tasks_list:
        print("错误：缺少必要的评分数据、编码器列表或STS任务列表。") # Console message in Chinese
        return
    if len(encoder_list) == 0:
        print("错误：编码器列表为空。") # Console message in Chinese
        return

    num_encoders = len(encoder_list)
    num_tasks = len(sts_tasks_list)

    x_encoder_positions = np.arange(num_encoders)  # X-axis positions for encoders

    # Define colors for different STS tasks
    # Using tab10 colormap, colors will repeat if num_tasks > 10
    colors = plt.cm.get_cmap('tab10', num_tasks if num_tasks <= 10 else 10) 

    bar_width_total_for_group = 0.8 # Total width for each group of bars (for one encoder)
    bar_width = bar_width_total_for_group / num_tasks # Width of a single bar

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10 + num_encoders * 1.5, 14)) # Adjust figure size

    # --- Plot Pearson Correlation ---
    ax1.set_title(f'Overall Pearson Correlation Performance on STS Tasks ({model_name})', fontsize=14) # Title in English
    for i, task_name in enumerate(sts_tasks_list):
        # Extract Pearson scores for the current task across all encoders (converted to percentage)
        task_pearson_scores = [scores_data[encoder]['pearson'][i] * 100 if scores_data[encoder]['pearson'] and i < len(scores_data[encoder]['pearson']) else 0 for encoder in encoder_list]
        
        # Calculate offset for the current task's bars within each encoder group
        offset = (i - num_tasks / 2 + 0.5) * bar_width
        
        rects = ax1.bar(x_encoder_positions + offset, task_pearson_scores, bar_width, label=task_name, color=colors(i % 10))
        ax1.bar_label(rects, padding=3, fmt='%.1f') # Display value on top of bar (already percentage)

    ax1.set_ylabel('Pearson Correlation (%)', fontsize=12) # Y-axis label in English
    ax1.set_xticks(x_encoder_positions)
    ax1.set_xticklabels(encoder_list, fontsize=12)
    ax1.legend(title="STS Task", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) # Legend title in English
    ax1.set_ylim(0, 105) # Y-axis range 0-100% (a bit more space for labels)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot Spearman Correlation ---
    ax2.set_title(f'Overall Spearman Correlation Performance on STS Tasks ({model_name})', fontsize=14) # Title in English
    for i, task_name in enumerate(sts_tasks_list):
        # Extract Spearman scores for the current task across all encoders (converted to percentage)
        task_spearman_scores = [scores_data[encoder]['spearman'][i] * 100 if scores_data[encoder]['spearman'] and i < len(scores_data[encoder]['spearman']) else 0 for encoder in encoder_list]
        
        offset = (i - num_tasks / 2 + 0.5) * bar_width
        
        rects = ax2.bar(x_encoder_positions + offset, task_spearman_scores, bar_width, label=task_name, color=colors(i % 10))
        ax2.bar_label(rects, padding=3, fmt='%.1f')

    ax2.set_ylabel('Spearman Correlation (%)', fontsize=12) # Y-axis label in English
    ax2.set_xticks(x_encoder_positions)
    ax2.set_xticklabels(encoder_list, fontsize=12)
    ax2.legend(title="STS Task", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) # Legend title in English
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to prevent legend overlap
    
    # Save the plot
    plt.savefig(output_image_file)
    print(f"图像已保存为 {output_image_file}") # Console message in Chinese
    plt.show()

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
RESULTS_SUBDIR = "results"
results_dir_path = os.path.join(script_dir, RESULTS_SUBDIR)

STS_TASKS_TO_PLOT = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
ENCODERS_TO_COMPARE = ['PromptEOL', 'DistillCore'] # Encoders on X-axis
MODEL_USED = "Qwen/Qwen2-7B-Instruct"
OUTPUT_IMAGE_FILENAME = "STS_Comparison_by_Encoder_EN.png" # Changed filename for English version

# --- Main script logic ---
if __name__ == "__main__":
    collected_scores_for_plot = {}
    for encoder in ENCODERS_TO_COMPARE:
        collected_scores_for_plot[encoder] = {'pearson': [], 'spearman': []}

    all_data_loaded_successfully = True
    expected_scores_per_encoder = len(STS_TASKS_TO_PLOT)

    for encoder_name in ENCODERS_TO_COMPARE:
        pearson_scores_for_encoder = []
        spearman_scores_for_encoder = []
        print(f"\n处理编码器: {encoder_name}") # Console message in Chinese
        for task_slug in STS_TASKS_TO_PLOT:
            filename = f"{encoder_name}_{task_slug}.json"
            filepath = os.path.join(results_dir_path, filename)
            
            print(f"  加载 {task_slug} 数据，来源: {filepath}...") # Console message in Chinese
            data = load_json_data(filepath)

            pearson_found_for_task = False
            spearman_found_for_task = False

            if data and task_slug in data:
                try:
                    pearson_all = data[task_slug]['all']['pearson']['all']
                    pearson_scores_for_encoder.append(pearson_all)
                    print(f"    {task_slug} - Pearson: {pearson_all:.3f}") # Console message in Chinese
                    pearson_found_for_task = True
                except (KeyError, TypeError) as e:
                    print(f"    错误: 在 {filename} 的 {task_slug} 任务中获取Pearson得分时缺少键或数据格式错误 ({e})。为此任务附加0。") # Console message in Chinese
                
                try:
                    spearman_all = data[task_slug]['all']['spearman']['all']
                    spearman_scores_for_encoder.append(spearman_all)
                    print(f"    {task_slug} - Spearman: {spearman_all:.3f}") # Console message in Chinese
                    spearman_found_for_task = True
                except (KeyError, TypeError) as e:
                    print(f"    错误: 在 {filename} 的 {task_slug} 任务中获取Spearman得分时缺少键或数据格式错误 ({e})。为此任务附加0。") # Console message in Chinese
            
            if not (data and task_slug in data): # File doesn't exist or task not in file
                 print(f"    警告: 未能加载文件 {filename} 或在文件中找不到任务 '{task_slug}'。为此任务附加0。") # Console message in Chinese
            
            if not pearson_found_for_task:
                pearson_scores_for_encoder.append(0) # Ensure placeholder if error occurred
            if not spearman_found_for_task:
                spearman_scores_for_encoder.append(0) # Ensure placeholder if error occurred
            
            if not (pearson_found_for_task and spearman_found_for_task):
                 all_data_loaded_successfully = False


        collected_scores_for_plot[encoder_name]['pearson'] = pearson_scores_for_encoder
        collected_scores_for_plot[encoder_name]['spearman'] = spearman_scores_for_encoder

    # Check data consistency
    consistent_data = True
    for encoder in ENCODERS_TO_COMPARE:
        if len(collected_scores_for_plot[encoder]['pearson']) != expected_scores_per_encoder or \
           len(collected_scores_for_plot[encoder]['spearman']) != expected_scores_per_encoder:
            print(f"错误: 编码器 {encoder} 收集到的分数数量不一致。预期 {expected_scores_per_encoder} 个任务的分数。") # Console message in Chinese
            consistent_data = False
            break
            
    if not all_data_loaded_successfully:
        print("\n注意: 部分数据点缺失或无法加载。绘图时将对这些缺失值使用0。") # Console message in Chinese

    if consistent_data:
        user_proceed = True
        if not all_data_loaded_successfully:
            user_input = input("部分数据缺失，是否继续使用0作为缺失值进行绘图? (yes/no): ").lower() # Console message in Chinese
            if user_input != 'yes':
                user_proceed = False
        
        if user_proceed:
            plot_overall_sts_comparison(
                ENCODERS_TO_COMPARE, # X-axis is encoders
                STS_TASKS_TO_PLOT,   # Bars represent these tasks
                collected_scores_for_plot,
                model_name=MODEL_USED,
                output_image_file=OUTPUT_IMAGE_FILENAME
            )
        else:
            print("由于数据缺失，用户中止绘图。") # Console message in Chinese
            
    else: # not consistent_data
        print("由于数据长度不一致，无法生成图像。") # Console message in Chinese

