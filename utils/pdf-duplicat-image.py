from PyPDF2 import PdfReader
from PIL import Image
import imagehash
import io
from collections import defaultdict
import os

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF and return their hashes."""
    reader = PdfReader(pdf_path)
    image_info = []

    for page_number, page in enumerate(reader.pages, 1):
        if '/XObject' in page['/Resources']:
            x_objects = page['/Resources']['/XObject'].get_object()
            for obj_name, obj in x_objects.items():
                obj = obj.get_object()  # Dereference the IndirectObject
                
                if obj['/Subtype'] == '/Image':
                    # Get image data
                    img_data = obj.get_data()
                    image_stream = io.BytesIO(img_data)
                    image = Image.open(image_stream)

                    # Calculate perceptual hash
                    image_hash = str(imagehash.average_hash(image))

                    image_info.append({
                        'page_number': page_number,
                        'image_name': obj_name,
                        'hash': image_hash,
                        'image': image
                    })

    return image_info

def find_duplicates(image_info):
    """Find duplicate images based on their perceptual hashes."""
    hash_groups = defaultdict(list)

    for info in image_info:
        hash_groups[info['hash']].append(info)

    # Filter only groups with duplicates
    duplicates = {k: v for k, v in hash_groups.items() if len(v) > 1}
    return duplicates

def report_duplicates(duplicates):
    """Generate a report of duplicate images."""
    if not duplicates:
        return "No duplicate images found."

    report = []
    for img_hash, instances in duplicates.items():
        group = [f"Page {info['page_number']} (Image Name: {info['image_name']})"
                 for info in instances]
        report.append(f"\nDuplicate group found in:")
        report.extend([f"  - {location}" for location in group])

    return "\n".join(report)

def save_unique_images(duplicates, image_info, output_dir):
    """Save one copy of each unique image."""
    os.makedirs(output_dir, exist_ok=True)

    # Create set of unique hashes
    unique_hashes = set()
    for info in image_info:
        if info['hash'] not in unique_hashes:
            unique_hashes.add(info['hash'])
            # Save the image
            output_path = os.path.join(output_dir, f"unique_image_{len(unique_hashes)}.png")
            info['image'].save(output_path)

def main(pdf_path, output_dir="unique_images"):
    """Main function to process PDF and find duplicate images."""
    image_info = extract_images_from_pdf(pdf_path)
    duplicates = find_duplicates(image_info)

    # Generate and print report
    report = report_duplicates(duplicates)
    print(report)

    # Save unique images
    save_unique_images(duplicates, image_info, output_dir)
    print(f"\nUnique images saved to: {output_dir}")

if __name__ == "__main__":
    main("1 TB Education VDH.pdf")
