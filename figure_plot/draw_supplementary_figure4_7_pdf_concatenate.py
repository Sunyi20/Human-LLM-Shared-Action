import os
from pypdf import PdfReader, PdfWriter

def merge_pdfs_side_by_side(wordcloud_dir, legend_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    wordcloud_files = sorted(os.listdir(wordcloud_dir))
    for wordcloud_filename in wordcloud_files:
        if not wordcloud_filename.endswith('.pdf'):
            continue

        dim_number = wordcloud_filename.split('_')[2]
            
        legend_filename = f"dim_{dim_number}"
        output_filename = f"merged_dim_{dim_number}"

        wordcloud_filepath = os.path.join(wordcloud_dir, wordcloud_filename)
        legend_filepath = os.path.join(legend_dir, legend_filename)
        output_filepath = os.path.join(output_dir, output_filename)

        reader_left = PdfReader(wordcloud_filepath)
        page_left = reader_left.pages[0]
        reader_right = PdfReader(legend_filepath)
        page_right = reader_right.pages[0]

        writer = PdfWriter()

        left_width = page_left.mediabox.width
        left_height = page_left.mediabox.height
        right_width = page_right.mediabox.width
        right_height = page_right.mediabox.height

        total_width = left_width + right_width
        max_height = max(left_height, right_height)


        new_page = writer.add_blank_page(width=total_width, height=max_height)

        left_offset_y = (max_height - left_height) / 2
        new_page.merge_translated_page(page_left, tx=0, ty=float(left_offset_y))

        right_offset_y = (max_height - right_height) / 2
        new_page.merge_translated_page(page_right, tx=float(left_width), ty=float(right_offset_y))

        with open(output_filepath, "wb") as f:
            writer.write(f)
            
base_dir = 'data/dim_visualization'

wordcloud_folder = os.path.join(base_dir, 'wordclouds_MLLM_qwen_72B')
legend_folder = os.path.join(base_dir, 'dim_visualization_MLLM_qwen_72B')

output_folder = os.path.join(base_dir, 'merged_MLLM_qwen_72B')
merge_pdfs_side_by_side(wordcloud_folder, legend_folder, output_folder)