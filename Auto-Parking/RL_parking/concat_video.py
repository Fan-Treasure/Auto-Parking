import moviepy.editor as mpy
import os


mode = '6'
input_dir = f'./log/training/任务{mode}训练视频'
output_dir = input_dir
output_file_path = os.path.join(output_dir, f'任务{mode}训练视频.mp4')

l = []
# for i in range(1000, 1001001, 100000):
for i in range(10000, 4010001, 400000):
    filepath = os.path.join(input_dir, f'DQN_{mode}_{i}.mp4')
    video = mpy.VideoFileClip(filepath).subclip(0.03)
    l.append(video)

final_clip = mpy.concatenate_videoclips(l)
final_clip.write_videofile(output_file_path)
