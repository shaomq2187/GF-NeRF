import os
import cv2
from tqdm import tqdm
from IPython import embed

def gen_video(image_folder):   
    image_names = os.listdir(image_folder)
    image_names.sort()
    image_folder_ori = '/'.join(image_folder.split('/')[:-1])
    video_path = os.path.join(image_folder_ori, "output.mp4")
    print(video_path)
    img = cv2.imread(os.path.join(image_folder, image_names[0]))
    size = img.shape[:2][::-1]
    fps = 10

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_path, fourcc, fps, size)  # 写入视频
    
    for image_name in tqdm(image_names[200:]):
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        video.write(img)
    
    video.release()
    # cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # image_folder = '/data/bearbee/aigc/aigc3d/nerfstudio/renders/demo_soho_output_camera3_depth/images'
    # gen_video(image_folder)

    image_folder = '/data/bearbee/aigc/aigc3d/nerfstudio/renders/temp_video_1683344977371/2023-05-08_195627_2/images'
    gen_video(image_folder)