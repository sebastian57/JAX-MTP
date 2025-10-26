#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import io
from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def find_run_files(target_dir, run_name):
    """Finds and organizes loss, test, and timing PDF files for a specific run."""
    level_files = {}
    timing_pdf_path = None
    run_info_file = None

    loss_pattern = re.compile(rf"loss_{re.escape(run_name)}_(\d+)\.pdf")
    timing_pdf_pattern = re.compile(rf"timing_{re.escape(run_name)}\.pdf")
    run_info_pattern = re.compile(rf"{re.escape(run_name)}\.txt")
    test_pattern = re.compile(rf"test_{re.escape(run_name)}_(\d+)\.pdf")
    print(f"Searching for test files like: test_{run_name}_<number>.pdf")
    print(f"Searching for loss files like: loss_{run_name}_<number>.pdf")
    print(f"Searching for timing file: timing_{run_name}.pdf")
    print(f"Searching for info file: {run_name}.txt")

    all_files = glob.glob(os.path.join(target_dir, "*"))

    for f_path in all_files:
        basename = os.path.basename(f_path)
        level = None 

        loss_match = loss_pattern.match(basename)
        if loss_match:
            level = int(loss_match.group(1))
            if level not in level_files:
                level_files[level] = {}
            level_files[level]['loss_pdf'] = f_path 
            print(f"  Found loss file: {basename} for level {level}")
            continue

        if test_pattern:
            test_match = test_pattern.match(basename)
            if test_match:
                level = int(test_match.group(1))
                if level not in level_files:
                    level_files[level] = {}
                level_files[level]['test_pdf'] = f_path 
                print(f"  Found test file: {basename} for level {level}")
                continue

        if timing_pdf_pattern.match(basename):
            timing_pdf_path = f_path
            print(f"  Found timing file: {basename}")
            continue

        if run_info_pattern.match(basename) and \
           not timing_pdf_pattern.match(basename) and \
           not loss_pattern.match(basename):
             if test_pattern and test_pattern.match(basename):
                  continue
             run_info_file = f_path
             print(f"  Found run info file: {basename}")
             continue

    sorted_levels = sorted(level_files.keys())
    return sorted_levels, level_files, timing_pdf_path, run_info_file


def draw_title_on_canvas(canvas_obj, title_text, text_style, x, y, width, height):
    """Draws title text onto an existing ReportLab canvas object."""
    p = Paragraph(title_text, text_style)
    w_avail = width
    h_avail = height
    w_req, h_req = p.wrapOn(canvas_obj, w_avail, h_avail)

    draw_x = x + (width - w_req) / 2
    draw_y = y - h_req
    p.drawOn(canvas_obj, draw_x, draw_y)
    
    
def merge_plot_onto_page(page_obj, plot_pdf_path, target_x, target_y, target_width, target_height):
    """Reads a plot PDF and merges it scaled and centered onto a target area."""
    if not (plot_pdf_path and os.path.exists(plot_pdf_path)):
        print(f"  Plot PDF not found or doesn't exist: {plot_pdf_path}")
        return False

    try:
        plot_reader = PdfReader(plot_pdf_path)
        if not plot_reader.pages:
             print(f"  Warning: Plot PDF '{plot_pdf_path}' has no pages.")
             return False
        plot_page = plot_reader.pages[0]

        plot_w = float(plot_page.mediabox.width)
        plot_h = float(plot_page.mediabox.height)

        if plot_w <= 0 or plot_h <= 0:
            print(f"  Warning: Invalid plot dimensions ({plot_w}x{plot_h}) in '{plot_pdf_path}'.")
            return False

        # Calculate scaling to fit plot within the target area
        scale_w = target_width / plot_w
        scale_h = target_height / plot_h
        scale_factor = min(scale_w, scale_h) # Use min to maintain aspect ratio and fit

        # Calculate translation to center the scaled plot within the target area
        scaled_w = plot_w * scale_factor
        scaled_h = plot_h * scale_factor
        translate_x = target_x + (target_width - scaled_w) / 2
        translate_y = target_y + (target_height - scaled_h) / 2 # Center vertically too

        op = Transformation().scale(sx=scale_factor, sy=scale_factor).translate(tx=translate_x, ty=translate_y)
        page_obj.merge_transformed_page(plot_page, op, expand=False)
        print(f"  Merged plot: {os.path.basename(plot_pdf_path)} into target area.")
        return True

    except Exception as e:
        print(f"  Error merging plot from {plot_pdf_path}: {e}")
        return False
                  
                  
                  
def combine_results_to_pdf(folder_name, run_name, output_filename,
                           threshold_loss, min_steps, lr_start,
                           transition_steps, decay_rate, global_norm_clip,
                           batch_size, min_dist, max_dist, scaling): 
    """
    Generates a combined PDF report placing a title and existing
    plot PDFs onto single pages.
    """
    base_output_dir = 'training_results'
    target_dir = os.path.join(base_output_dir, folder_name)

    if not os.path.isdir(target_dir):
        print(f"Error: Target directory '{target_dir}' does not exist.")
        return

    print(f"Scanning directory: {target_dir}")
    print(f"Looking for run name pattern: {run_name}")
    if run_name:
         print(f"Looking for test name pattern: {run_name}")


    # <<< MODIFIED: Pass run_name to find_run_files >>>
    sorted_levels, level_files, timing_pdf_path, run_info_file = find_run_files(target_dir, run_name)

    # --- Info Page Generation (Keep as is) ---
    pdf_writer = PdfWriter()
    styles = getSampleStyleSheet()
    title_style = styles['h1']
    info_title_style = styles['h2']
    normal_style = styles['Normal']
    page_width, page_height = letter
    margin = 0.75 * inch
    content_width = page_width - 2 * margin
    content_height = page_height - 2 * margin

    print("Generating Info Page...")
    # ... (Keep the entire Info Page try...except block exactly as it was) ...
    try:
        info_buffer = io.BytesIO()
        c_info = canvas.Canvas(info_buffer, pagesize=letter)
        y_position = page_height - margin
        title_text = f"Run Report: {run_name}"
        p_title = Paragraph(title_text, title_style)
        title_w, title_h = p_title.wrapOn(c_info, content_width, content_height)
        p_title.drawOn(c_info, margin + (content_width - title_w)/2, y_position - title_h)
        y_position -= (title_h + 0.3*inch)
        params_title_text = "Run Parameters:"
        p_params_title = Paragraph(params_title_text, info_title_style)
        pt_w, pt_h = p_params_title.wrapOn(c_info, content_width, content_height)
        p_params_title.drawOn(c_info, margin, y_position - pt_h)
        y_position -= (pt_h + 0.1*inch)
        info_lines = [
            f"<b>Folder:</b> {folder_name}", f"<b>Run Name:</b> {run_name}",
            f"<b>Test Name (if used):</b> {run_name if run_name else 'N/A'}", # <<< NEW Info Line
            f"<b>Convergence Threshold (Loss Diff):</b> {threshold_loss:.2e}",
            f"<b>Minimum Steps per Phase:</b> {min_steps}",
            f"<b>Initial Learning Rate:</b> {lr_start:.2e}",
            f"<b>Transition Steps:</b> {transition_steps}",
            f"<b>Decay Rate:</b> {decay_rate:.2e}",
            f"<b>Global Clipping Norm:</b> {global_norm_clip:.2e}",
            f"<b>Batch Size:</b> {batch_size:.2e}",
            f"<b>Min Dist:</b> {min_dist:.2e}",
            f"<b>Max Dist:</b> {max_dist:.2e}",
            f"<b>Scaling:</b> {scaling:.2e}"

        ]
        for line in info_lines:
            p_info = Paragraph(line, normal_style)
            info_w, info_h = p_info.wrapOn(c_info, content_width, content_height)
            if y_position - info_h < margin: break
            p_info.drawOn(c_info, margin, y_position - info_h)
            y_position -= (info_h + 0.05*inch)
        c_info.save()
        info_buffer.seek(0)
        info_reader = PdfReader(info_buffer)
        if info_reader.pages:
             info_page = info_reader.pages[0]
             pdf_writer.add_page(info_page)
             print("  Info Page added successfully.")
        else: print("  Warning: Failed to create info page.")
    except Exception as e: print(f"  Error generating info page: {e}")
    # --- End Info Page ---


    # --- Define Layout Areas for Level Pages ---
    title_area_height = 0.75 * inch
    title_area_y = page_height - margin
    title_area_width = page_width - 2 * margin

    # Area below title for plots
    plots_total_area_height = page_height - (2 * margin) - title_area_height
    plots_total_area_y = margin
    plots_total_area_width = page_width - 2 * margin

    # <<< NEW: Calculate dimensions for two side-by-side plots >>>
    plot_gap = 0.2 * inch # Gap between the two plots
    plot_width_per_item = (plots_total_area_width - plot_gap) / 2
    plot_height_per_item = plots_total_area_height # Each plot uses the full height

    # X coordinates for the start of each plot's target area
    plot1_x = margin
    plot2_x = margin + plot_width_per_item + plot_gap
    # Y coordinate for the bottom of each plot's target area
    plot_y_bottom = plots_total_area_y

    # --- Loop through Levels ---
    print("\nProcessing Level Plots...")
    if not sorted_levels:
        print("  No level-specific loss plots found.")
    else:
        for level in sorted_levels:
            print(f"Processing Level {level}...")
            level_data_paths = level_files.get(level, {})
            loss_pdf_path = level_data_paths.get('loss_pdf') # <<< MODIFIED: Use specific key
            test_pdf_path = level_data_paths.get('test_pdf') # <<< NEW: Get test plot path

            # Check if at least one plot exists for this level
            if not loss_pdf_path and not (run_name and test_pdf_path):
                print(f"  Warning: No loss or test plot PDF found for level {level}. Skipping this page.")
                continue

            try:
                # Create a blank page for this level
                blank_page = PageObject.create_blank_page(width=page_width, height=page_height)

                # <<< MODIFIED: Use helper function to merge plots >>>
                # Merge Loss Plot (Plot 1 - Left side)
                if loss_pdf_path:
                    print("  Attempting to merge loss plot...")
                    merge_plot_onto_page(
                        blank_page,
                        loss_pdf_path,
                        plot1_x,
                        plot_y_bottom,
                        plot_width_per_item,
                        plot_height_per_item
                    )
                else:
                    print("  Loss plot PDF not found for this level.")

                # Merge Test Plot (Plot 2 - Right side, only if run_name provided)
                if run_name and test_pdf_path:
                    print("  Attempting to merge test plot...")
                    merge_plot_onto_page(
                        blank_page,
                        test_pdf_path,
                        plot2_x,
                        plot_y_bottom,
                        plot_width_per_item,
                        plot_height_per_item
                    )
                elif run_name:
                    print(f"  Test plot PDF (using run_name '{run_name}') not found for this level.")


                # <<< MODIFIED: Create and merge title overlay (stays the same) >>>
                title_text = f"Level {level} Results"
                title_buffer = io.BytesIO()
                c_title = canvas.Canvas(title_buffer, pagesize=(page_width, page_height))
                # Use specific coordinates for title area
                draw_title_on_canvas(c_title, title_text, styles['h2'], # Use H2 for level titles
                                     margin, title_area_y,
                                     title_area_width, title_area_height)
                c_title.save()
                title_buffer.seek(0)
                title_reader = PdfReader(title_buffer)
                if title_reader.pages:
                     title_page_only = title_reader.pages[0]
                     blank_page.merge_page(title_page_only) # Merge title overlay
                     print(f"  Added title: '{title_text}'")
                else:
                     print("  Warning: Failed to create title page overlay.")

                # Add the completed page to the writer
                pdf_writer.add_page(blank_page)

            except Exception as e:
                print(f"  Error processing level {level}: {e}")

    # --- Timing Plot Processing (Keep as is, assuming full width) ---
    print("\nProcessing Timing Plot...")
    if timing_pdf_path and os.path.exists(timing_pdf_path):
        try:
            blank_page_timing = PageObject.create_blank_page(width=page_width, height=page_height)
            # Use the original plot area dimensions for the timing plot
            timing_plot_target_width = plots_total_area_width
            timing_plot_target_height = plots_total_area_height
            timing_plot_target_x = margin
            timing_plot_target_y = plots_total_area_y

            # Use the helper function
            merged = merge_plot_onto_page(
                blank_page_timing,
                timing_pdf_path,
                timing_plot_target_x,
                timing_plot_target_y,
                timing_plot_target_width,
                timing_plot_target_height
            )

            if merged:
                # Add Title Overlay
                title_text_t = "Overall Timing Results"
                title_buffer_t = io.BytesIO()
                c_t = canvas.Canvas(title_buffer_t, pagesize=(page_width, page_height))
                draw_title_on_canvas(c_t, title_text_t, styles['h2'], margin, title_area_y, title_area_width, title_area_height)
                c_t.save()
                title_buffer_t.seek(0)
                title_reader_t = PdfReader(title_buffer_t)
                if title_reader_t.pages:
                     title_page_only_t = title_reader_t.pages[0]
                     blank_page_timing.merge_page(title_page_only_t)
                     print(f"  Added title: '{title_text_t}'")
                else:
                     print("  Warning: Failed to create timing title page overlay.")

                pdf_writer.add_page(blank_page_timing)
            # else: # Error message already printed by helper function

        except Exception as e:
            print(f"  Error processing timing files: {e}")
    else:
        print("  No timing plot found.")


    # --- Final PDF Writing (Keep as is) ---
    final_output_path = os.path.join(target_dir, f"{output_filename}.pdf") # Add .pdf extension directly
    # ... (Keep the final PDF writing try...except block exactly as it was) ...
    try:
        if len(pdf_writer.pages) > 0:
            pdf_writer.add_metadata({ '/Producer': 'combine_pdfs.py script', '/Title': f'Combined Report for {run_name}'})
            with open(final_output_path, "wb") as f_out: pdf_writer.write(f_out)
            print(f"\nSuccessfully created combined report: {final_output_path}")
        else: print("\nWarning: No pages were added to the PDF writer. Output file not created.")
    except Exception as e:
        print(f"\nError writing final PDF '{final_output_path}': {e}")
        if os.path.exists(final_output_path):
             try: os.remove(final_output_path); print(f"  Deleted potentially corrupted output file.")
             except OSError as oe: print(f"  Error trying to delete corrupted file: {oe}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine PDF plots and TXT data from a training run into a single report PDF."
    )

    parser = argparse.ArgumentParser(description="Run MTP Jax training timing test.")
    parser.add_argument("--folder_name", type=str, default=None,
                        help="String of the name of the input folder")
    parser.add_argument("--run_name", type=str, default=None,
                        help="String of the run name")
    parser.add_argument("--output_name", type=str, default="combined_report",
                        help="String of the name of the output file")
    parser.add_argument("--threshold_loss", type=float, default=1e-5,
                    help="Convergence criterium value used (loss diff <= threshold)")
    parser.add_argument("--min_steps", type=int, default=10,
                    help="Minimum steps per phase used before convergence check")
    parser.add_argument("--lr_start", type=float, default=1e-2,
                    help="Initial learning rate value used")
    parser.add_argument("--transition_steps", type=int, default=1,
                        help="After how many steps does the learning rate decay")
    parser.add_argument("--decay_rate", type=float, default=0.99,
                        help="Decay rate per 1 times transition steps")
    parser.add_argument("--global_norm_clip", type=float, default=1e-1,
                        help="Value for clipping gradients")
    parser.add_argument("--scaling", type=float, default=1.0,
                        help="rescaling factor for the mtp (should do nothing here tbh)")
    parser.add_argument("--max_dist", type=float, default=5.0,
                        help="Maximum interaction distance for the mtp")
    parser.add_argument("--min_dist", type=float, default=0.5,
                        help="Minimum interaction distance for the mtp")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Size of the individual batches used for mini batching")
    args = parser.parse_args()

    combine_results_to_pdf(args.folder_name, args.run_name, args.output_name, args.threshold_loss, args.min_steps, args.lr_start, 
                           args.transition_steps, args.decay_rate, args.global_norm_clip, args.batch_size, args.min_dist, args.max_dist, args.scaling)
