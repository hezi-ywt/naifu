import os
import json

def isimage(file):
    end = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]
    for i in end:
        if file.endswith(i):
            return True
    return False


def make_json_by_dir(dirname, json_pth, json_in_pth=None):
    if json_in_pth:
        try:
            with open(json_in_pth, "r") as f:
                js = json.load(f)
        except:
            js = []
            print("json_in_pth error!")
    else:
        js = []

    for i in os.listdir(dirname):

        image_path = os.path.join(dirname, i)
        if isimage(image_path):
            txt_path = os.path.join(dirname, i.split(".")[0] + ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    text = f.read()
            else:
                text = ""
            kwarg = {
                "tag_artist": "",
                "tag_character": "",
                "tag_general": "",
                "tag_meta": "",
                
            }
            js.append({"image_path": image_path, "text_zh": text, "kwarg": {}})

    with open(json_pth, "w") as f:
        json.dump(js, f)

    print(len(js))

def make_pid_json_by_dir(dirname, json_pth,json_in_pth=None):
    if json_in_pth:
        try:
            with open(json_in_pth, "r") as f:
                js = json.load(f)
        except:
            js = []
            print("json_in_pth error!")
    else:
        js = []

    for i in os.listdir(dirname):

        image_path = os.path.join(dirname, i)
        if isimage(image_path):
            kwarg = {

                    
                }
            txt_path = os.path.join(dirname, i.split(".")[0] + ".txt")
            txt_cog_path = os.path.join(dirname, i.split(".")[0] + "_cog.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    text = f.read()
                    text = text.replace(", general, sensitive, questionable, explicit", "")
                    text = text.replace("|||", "")
                    text = text.replace(" (role)", "")
                    text = text.replace(" (style)", "")
                kwarg["text_zh1"] = text
                
            if os.path.exists(txt_cog_path):
                with open(txt_cog_path, "r") as f:
                    text_cog = f.read()
                    text_cog = text_cog.replace(", general, sensitive, questionable, explicit", "")
                    text_cog = text_cog.replace("|||", "")
                text_zh = text_cog
            else:
                text_zh = ""
                continue
            
            
            tag_artist = ""
            tag_character = ""
            tag_general = ""
            rating_tag = ""
            tag_meta = ""
            year = ""
            quality = ""
            copyright_tag = ""
            cn = {}
            caption = {}
            text_zh1 = ""
            text_zh2 = ""
            
            
            tostr = [tag_artist, tag_character, tag_general, year, quality, copyright_tag, rating_tag]
            
            meta_blacklist = ["commission","skeb commission","pixiv commission", "(medium)",
                            "commentary",
                            "bad",
                            "translat",
                            "request",
                            "mismatch",
                            "revision",
                            "audio",
                            "commission"
                            "video",
                            "paid reward", 
                            "patreon reward",
                            
                            ]
            for i in tostr:
                if not isinstance(i, str):
                    if isinstance(i, list):
                        
                        i = ", ".join(i)
                        
                    else:
                        print(f"type error")


            if isinstance(tag_meta, str):
                tag_meta = tag_meta.split(", ")

            for i in meta_blacklist:
                if i in tag_meta:
                    tag_meta.remove(i)
                    
            tag_meta = ", ".join(tag_meta)
                
            kwarg = {
                "tag_artist": tag_artist,
                "tag_character": tag_character,
                "tag_general": tag_general,
                "tag_meta": tag_meta,
                "tag_copyright":copyright_tag,
                "year_tag": year,
                "rating_tag": rating_tag,
                "quality_tag": quality,
                "text_zh1": text_zh1,
                "text_zh2": text_zh2,
                "character_zh": cn,
                
            }
            js.append({"image_path": image_path, "text_zh": text_zh, "kwarg": kwarg})

    with open(json_pth, "w") as f:
        json.dump(js, f)

    print(len(js))
    
    
    
def make_json_by_dir_and_meta_pid(dirname, json_pth, meta_json={}):
    js = []
    for i in os.listdir(dirname):

        image_path = os.path.join(dirname, i)
        
        if isimage(image_path):
            pid = int(i.split(".")[0])
            
            tag_artist = meta_json[str(pid)].get("artist", "")
            tag_character = meta_json[str(pid)].get("tag_character", "")
            tag_general = meta_json[str(pid)].get("tag_general", "")
            rating_tag = meta_json[str(pid)].get("rating_tag", "")
            tag_meta = meta_json[str(pid)].get("meta", "")
            
            year = meta_json[str(pid)].get("year_tag", "")
            quality = meta_json[str(pid)].get("quality_tag", "")
            copyright_tag = meta_json[str(pid)].get("copy_right", "")
            
            
            cn = meta_json[str(pid)].get("cn", {})
            caption = meta_json[str(pid)].get("caption", {})
            text_zh = caption.get("caption_round1_base", "")
            text_zh1 = caption.get("caption_round2_overlap", "")
            text_zh2 = caption.get("caption_round3_overlap", "")
            
            
            tostr = [tag_artist, tag_character, tag_general, year, quality, copyright_tag, rating_tag]
            
            meta_blacklist = ["commission","skeb commission","pixiv commission", "(medium)",
                            "commentary",
                            "bad",
                            "translat",
                            "request",
                            "mismatch",
                            "revision",
                            "audio",
                            "commission"
                            "video",
                            "paid reward", 
                            "patreon reward",
                            
                            ]
            for i in tostr:
                if not isinstance(i, str):
                    if isinstance(i, list):
                        
                        i = ", ".join(i)
                        
                    else:
                        print(f"type error")


            if isinstance(tag_meta, str):
                tag_meta = tag_meta.split(", ")

            for i in meta_blacklist:
                if i in tag_meta:
                    tag_meta.remove(i)
                    
            tag_meta = ", ".join(tag_meta)
                
            kwarg = {
                "tag_artist": tag_artist,
                "tag_character": tag_character,
                "tag_general": tag_general,
                "tag_meta": tag_meta,
                "tag_copyright":copyright_tag,
                "year_tag": year,
                "rating_tag": rating_tag,
                "quality_tag": quality,
                "text_zh1": text_zh1,
                "text_zh2": text_zh2,
                "character_zh": cn,
                
            }
            js.append({"image_path": image_path, "text_zh": text_zh, "kwarg": kwarg})

    with open(json_pth, "w") as f:
        json.dump(js, f)

    print(len(js))


def make_json_by_dir_and_meta(dirname, json_pth, meta_json={}):
    js = []
    for i in os.listdir(dirname):

        image_path = os.path.join(dirname, i)
        
        if isimage(image_path):
            name = (i.split(".")[0])
            text_file = os.path.join(dirname, name + ".txt")
            if os.path.exists(text_file):
                with open(text_file, "r") as f:
                    text = f.read()
                    
                    
            
            
            
            
            tag_artist = ""
            tag_character = ""
            tag_general = ""
            rating_tag = ""
            tag_meta = ""
            
            year = ""
            quality = ""
            copyright_tag = ""
            
            
            cn = {}
            caption = {}
            text_zh = text
            text_zh1 = ""
            text_zh2 = ""
            
            
            tostr = [tag_artist, tag_character, tag_general, year, quality, copyright_tag, rating_tag]
            
            meta_blacklist = ["commission","skeb commission","pixiv commission", "(medium)",
                            "commentary",
                            "bad",
                            "translat",
                            "request",
                            "mismatch",
                            "revision",
                            "audio",
                            "commission"
                            "video",
                            "paid reward", 
                            "patreon reward",
                            
                            ]
            for i in tostr:
                if not isinstance(i, str):
                    if isinstance(i, list):
                        
                        i = ", ".join(i)
                        
                    else:
                        print(f"type error")


            if isinstance(tag_meta, str):
                tag_meta = tag_meta.split(", ")

            for i in meta_blacklist:
                if i in tag_meta:
                    tag_meta.remove(i)
                    
            tag_meta = ", ".join(tag_meta)
                
            kwarg = {
                "tag_artist": tag_artist,
                "tag_character": tag_character,
                "tag_general": tag_general,
                "tag_meta": tag_meta,
                "tag_copyright":copyright_tag,
                "year_tag": year,
                "rating_tag": rating_tag,
                "quality_tag": quality,
                "text_zh1": text_zh1,
                "text_zh2": text_zh2,
                "character_zh": cn,
                
            }
            js.append({"image_path": image_path, "text_zh": text_zh, "kwarg": kwarg})

    with open(json_pth, "w") as f:
        json.dump(js, f)

    print(len(js))
    
if __name__ == "__main__":
    # dirs = ["/home1/qbs/my_program1/data1/onee-shota", "/home1/qbs/my_program1/data1/breastfeeding","/home1/qbs/my_program1/data/upside-down","/home1/qbs/my_program1/data/hand_focus","/home1/qbs/my_program1/data/foot_focus","/home1/qbs/my_program1/data/fingering","/home1/qbs/my_program1/data/cli_out_o","/home1/qbs/my_program1/data/giantess"]

    # json_pth = "/home1/qbs/my_program1/HunyuanDiT/dataset/hy_data.json"
    
    # for dirname in dirs:
    #     # make_json_by_dir(dirname, json_pth,json_in_pth=json_pth)
    #     make_pid_json_by_dir(dirname, json_pth,json_in_pth=json_pth)
    
    # print("Done!")
    
    dir = "/data/HyData/caption_data/data_hun_compressed"
    
    meta_json_pth = "/root/downloaded_models/HunyuanDiT_v1.2_danbooru_710k/caption.json"
    
    
    with open(meta_json_pth, "r") as f:
        meta_json = json.load(f)    
        
    # make_json_by_dir_and_meta_pid(dir, json_pth, meta_json = meta_json)
    
    Dir = "/data/HyData_out"
    # for dir in os.listdir(Dir):
    #     if os.path.isdir(os.path.join(Dir, dir)):
    #         json_pth = f"{Dir}/{dir}.json"
    #         make_json_by_dir_and_meta_pid(os.path.join(Dir, dir), json_pth, meta_json = meta_json)
    # json_pth = "/root/HunyuanDiT/dataset/hy_caption_data.json"
    
    # make_pid_json_by_dir(dir, json_pth)
    
    