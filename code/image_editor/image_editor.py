import csv
import os
from turtle import width
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import piexif

"""
2025dec.

Creates uniform pictures, including additional text from csv file.
Needs:
* CSV file/table
* needs pictures in /image/
* exports pictures to /output/

+ add text at different locations.
- file metadata is not working.

"""


# --- Add metadata to picture file ---
def add_metadata_png(
    title=None,
    description=None,
    caption=None,
    author=None,
    copyright_text=None
):
    # ---------- PNG ----------
    meta = PngInfo()

    if title:
        meta.add_text("Title", title)
    if description:
        meta.add_text("Description", description)
    if caption:
        meta.add_text("Caption", caption)
    if author:
        meta.add_text("Author", author)
    if copyright_text:
        meta.add_text("Copyright", copyright_text)
    #img.save(output_path, pnginfo=meta)
    return meta

# --- Add metadata to picture file ---
def add_metadata_jpeg(
    title=None,
    description=None,
    caption=None,
    author=None,
    copyright_text=None
):
    # ---------- JPEG ----------
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    if description:
        exif_dict["0th"][piexif.ImageIFD.XPSubject] = description.encode("utf-16le")
    if title:
        exif_dict["0th"][piexif.ImageIFD.XPTitle] = title.encode("utf-16le")
    if author:
        exif_dict["0th"][piexif.ImageIFD.Artist] = author.encode("utf-8")
    if copyright_text:
        exif_dict["0th"][piexif.ImageIFD.Copyright] = copyright_text.encode("utf-8")
    if caption:
        exif_dict["0th"][piexif.ImageIFD.XPComment] = caption.encode("utf-16le")
    exif_bytes = piexif.dump(exif_dict)
    #img.save(output_path, exif=exif_bytes)
    return exif_bytes

    

# --- Add text on image ---
def add_text(
    image,
    position,
    text,
    text_color,
    font_path,
    font_size,
    alignment="left",
    line_spacing=4
):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    x, y = position

    # Measure multiline text
    bbox = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=line_spacing,
        align=alignment
    )

    text_width = bbox[2] - bbox[0]

    # Adjust x based on alignment
    if alignment == "center":
        x -= text_width // 2
    elif alignment == "right":
        x -= text_width

    # text = "may contain \n newline"
    draw.multiline_text(
        (x, y),
        text,
        fill=text_color,
        font=font,
        spacing=line_spacing,
        align=alignment
    )


# --- Configuration ---
#root = os.getcwd()
image_dir = r"d:/Github/LDraw/git/code/add_text_to_pictures/images"
output_dir = r"d:/Github/LDraw/git/code/add_text_to_pictures/output"
csv_path = r"d:/Github/LDraw/git/code/add_text_to_pictures/model_info.csv"

os.makedirs(output_dir, exist_ok=True)


# --- Read CSV & Update pictures ---
with open(csv_path, newline="", encoding="cp1252") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')

    for row in reader:
        filename = row["filename"]
        caption_txt = row["caption"].replace("\\n", "\n")   #need to replace '\n'
        nrofparts_txt = row["nrofparts"] 
        extrainfo_txt = row["extrainfo"]

        input_path = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.isfile(input_path):
            #print(f"Skipping (not found): {filename}")
            continue

        # --- Open image (keep transparency) ---
        img = Image.open(input_path).convert("RGBA")
        #draw = ImageDraw.Draw(img)        

        # --- Add text ---
        font_path = "arial.ttf"
        text_color1 = (32, 32, 32, 255)  #black
        #text_color1 = (255,255,255,255) #white
        font_size = 48
        margin = 40
        add_text(
            image=img,
            position=(img.width // 2, margin),
            text=caption_txt,
            text_color=text_color1,
            font_path=font_path,
            font_size=font_size,
            alignment="center"
        )

        font_size = 32
        line_height = 35
        margin = 30
        if len(extrainfo_txt)>0:
            add_text(
                img,
                (margin, img.height - margin - 2 * line_height),
                extrainfo_txt,
                text_color1,
                font_path,
                font_size=font_size,
                alignment="left"
            )        
        add_text(
            img,
            (margin, img.height - margin - line_height),
            nrofparts_txt,
            text_color1,
            font_path,
            font_size=font_size,
            alignment="left"
        )
        add_text(
            img,
            (img.width - margin, img.height - margin - line_height),
            "retrobuildingtoys.nl",
            text_color1,
            font_path,
            font_size=font_size,
            alignment="right"
        )
        
        # --- Save updated image ---
        #caption_txt.replace("\n", " ")   #need to replace '\n' to space
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".png":
            #title, description, caption, author, copyright_text
            meta = add_metadata_png(caption_txt, nrofparts_txt + "; " + extrainfo_txt, caption_txt, "www.retrobuildingtorys.nl", "© 2025"  )
            img.save(output_path, format="PNG", pnginfo=meta)
            print(f"Updated: {filename}")
            #print(Image.open(output_path).info)
        elif ext in (".jpg", ".jpeg"):
            img = img.convert("RGB")
            exif_bytes = add_metadata_jpeg(caption_txt, nrofparts_txt + "; " + extrainfo_txt, caption_txt, "www.retrobuildingtorys.nl", "© 2025" )
            img.save(output_path, format="JPEG", exif=exif_bytes) #quality=95, subsampling=0
            print(f"Updated: {filename}")
        else:
            img.save(output_path)
            print(f"Updated: {filename} (w/o metadata)")
        
