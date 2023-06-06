from PIL import Image
from optparse import OptionParser
from lang_sam import LangSAM
import torchvision
import sys
import cv2
import numpy as np

def main(options):
    model = LangSAM()
    image_pil = Image.open(options.image_path).convert("RGB")
    text_prompt = options.text_prompt
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    masks = masks * 255.0
    if options.save_mask:
        filename = f"./outputs/{options.image_path.split('.')[0]}_{options.text_prompt}.jpeg"
        torchvision.utils.save_image(masks, filename)
    if options.render_mask:
        image = masks.numpy().squeeze(0)
        cv2.imshow("masks_render", image)
        cv2.waitKey(0) # wait for ay key to exit window
        cv2.destroyAllWindows()  # close all windows



if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--image", dest="image_path", type="string",
                      default="rgb_static.jpeg",
                      help="Path to Image file")
    parser.add_option("--text", "--text_prompt", dest="text_prompt", type="string",
                      default="table",
                      help="Text prompt")
    parser.add_option("--save", action="store_true", dest="save_mask", default=True,
                      help="Flag to save the mask")
    
    parser.add_option("--render", action="store_true", dest="render_mask", default=True,
                      help="Flag to render the mask")
    
    (options, args) = parser.parse_args(sys.argv)
    main(options)

# CUDA_VISIBLE_DEVICES=0 python test.py -i rgb_static.jpeg --save --render --text "table"