from PIL import Image, ImageDraw, ImageFont

conformer_path = "/home/ish/rudn/VKR/reports/figures/summary/train_rul_hybrid_v5_odd/balanced/ws2048/conformer_train_rul_hybrid_v5_odd_profilebalanced_trials30_epochs25_featurecache_on_ws2048_importance.png"
patchtst_path = "/home/ish/rudn/VKR/reports/figures/summary/train_rul_hybrid_v5_odd/balanced/ws2048/patchtst_train_rul_hybrid_v5_odd_profilebalanced_trials30_epochs25_featurecache_on_ws2048_importance.png"

out_path = "/home/ish/rudn/VKR/reports/figures/v5_odd_optuna_importance_composite.png"

try:
    img1 = Image.open(conformer_path)
    img2 = Image.open(patchtst_path)
    
    # Calculate dimensions
    width = max(img1.width, img2.width)
    title_height = 80
    padding = 20
    height = img1.height + img2.height + title_height + padding * 3
    
    # Create new image with a light gray background
    new_img = Image.new('RGB', (width, height), color='#f4f4f9')
    
    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        font_sub = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        
    draw = ImageDraw.Draw(new_img)
    
    # Draw Title
    title = "Рисунок 2.13. Optuna parameter importance для v5_odd"
    # Fallback positioning since textbbox might fail with default font
    draw.text((width//2 - 300, 20), title, fill='black', font=font)
    
    # Paste Conformer
    new_img.paste(img1, (0, title_height + padding))
    
    # Paste PatchTST
    new_img.paste(img2, (0, title_height + img1.height + padding * 2))
    
    new_img.save(out_path)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
