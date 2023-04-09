from utils import * 
import sys
#add path to parent directory in sys.path list
sys.path.append('C:\\Users\\shash\\Desktop\\btp\\main\\model\\')
from layers.summarizer import CA_SUM
c,loaded_model,model = None,None,None 
from logger import logger
import os
import cv2
from pydub import AudioSegment
import tracemalloc as tm

def load_models():
    global c,loaded_model,model
    c = CA_SUM()
    loaded_model = torch.load('C:\\Users\\shash\\Desktop\\btp\\main\\trained_model_state\\SumMe_output.pt')
    c.load_state_dict(loaded_model)
    c.eval()
    PATH = 'C:\\Users\\shash\\Desktop\\btp\\main\\trained_model_state\\googlenet-feature-extractor.pt'
    model = torch.load(PATH)
    model.eval()
    set_models(c,loaded_model,model)

def generate_summary(id,file_name,summary_percentage=10):
    '''if logger.active:
        tm.start()'''
    logger.log((file_name,id,summary_percentage))
    base_path = 'C:\\Users\\shash\\Desktop\\btp\\backend\\static'
    base_name = file_name.strip(".mp4")
    video_path = f'{base_path}\\uploads\\{id}\\{file_name}'
    audio_path = f'{base_path}\\uploads\\{id}\\{base_name + ".mp3"}'
    os.system(f'ffmpeg -i "static/uploads/{id}/{file_name}" -q:a 0 -map a "static/uploads/{id}/{base_name}.mp3"')
    sound = AudioSegment.from_mp3(audio_path)
    actual_frames,audio_index = get_frames_with_sound(video_path)
    l=len(audio_index)
    i=l-1
    sl = len(sound)-1
    while audio_index[i] == 0 and i > 0:
        audio_index[i] = sl - (l-1-i)
        i -= 1
    v_len = len(actual_frames)
    features = extract_range_sync(actual_frames,0,v_len)
    '''snapshot1 = tm.take_snapshot()
    stats = snapshot1.statistics('lineno')
    with open("memory2.txt" , "w+") as f:
        f.write("\n".join(str(i) for i in stats))
        f.close()'''
    logger.log(f'exraction completed al={len(actual_frames)}, fl = {len(features)}')
    features = np.array(features)
    features = torch.from_numpy(features)
    frame_features = features.view(-1, 1024).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    _ = c.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        scores, attn_weights = _(frame_features)  # [1, seq_len]
        scores = scores.squeeze(0).cpu().numpy().tolist()
        attn_weights = attn_weights.cpu().numpy()
    no_of_shots = (len(scores) * summary_percentage) // 100 
    logger.log((no_of_shots,len(scores)))
    ids = get_video_shots(frame_features,max_shots=no_of_shots)
    shots_ids = []
    shot_scores = []
    frames_per_shot = []
    lid = len(ids)
    for i in range(lid - 1):
        #shots.append(actual_frames[ids[i] : ids[i+1]])
        shots_ids.append((ids[i] , ids[i+1]))
        frames_per_shot.append(ids[i+1]-ids[i])
        shot_scores.append(sum(scores[ids[i] : ids[i+1]]) / (ids[i+1] - ids[i]))
    #shots.append(actual_frames[ids[lid - 1]:])
    shots_ids.append((ids[lid - 1] , len(actual_frames)))
    shot_scores.append(sum(scores[ids[-1] :])/(len(scores) - ids[-1]))
    frames_per_shot.append(len(scores) - ids[-1])
    max_frames = (len(actual_frames) * summary_percentage) // 100 
    logger.log((max_frames, frames_per_shot, len(shot_scores),no_of_shots))
    selected  = knapSack(max_frames, frames_per_shot, shot_scores,no_of_shots)
    fh,fw,fl = actual_frames[0].shape
    size = (fw,fh)
    if not os.path.isdir(f'static/summaries/{id}'):
        os.mkdir(f'static/summaries/{id}')
    save_path = f'{base_path}\\summaries\\{id}\\{base_name}_temp.mp4'
    vidcap = cv2.VideoCapture(video_path)
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    cv2.destroyAllWindows()
    #cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path , 0,frame_rate,size)
    for i in selected :
        start_id = shots_ids[i][0]
        end_id = shots_ids[i][1]
        for frame in actual_frames[start_id : end_id]:
            out.write(frame)
    out.release()
    audio_save_path = f'{base_path}\\summaries\\{id}\\{base_name}.mp3'
    audio_segment = AudioSegment.empty()
    for i in selected:
        start_id = shots_ids[i][0]
        end_id = shots_ids[i][1]
        audio_segment += sound[audio_index[start_id] : audio_index[end_id]]
    audio_segment.export(audio_save_path)
    os.system(
        f'ffmpeg -i "static/summaries/{id}/{base_name}_temp.mp4"\
              -i "static/summaries/{id}/{base_name}.mp3" -c:v copy -c:a aac \
              "static/summaries/{id}/{file_name}" < in.txt')
    os.system(f'del "{base_path}\\summaries\\{id}\\{base_name}_temp.mp4"')
    os.system(f'del "{base_path}\\summaries\\{id}\\{base_name}.mp3"')
    os.system(f'del "{base_path}\\uploads\\{id}\\{base_name}.mp3"')
    print('End')
def setup():
    load_models()
