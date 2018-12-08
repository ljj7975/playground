import json
import librosa
import os
import re
import search
import string
import subprocess
from pytube import YouTube

video_dict = {}

def grab_videos(keyword, token=None):
	res = search.youtube_search(keyword, max_results=1)
	# res = search.youtube_search(keyword)
	token = res[0]
	videos = res[1]
	for video_data in videos:
		# print(json.dumps(video_data, indent=4, sort_keys=True))
		video_id = video_data['id']['videoId']
		video_title = video_data['snippet']['title']
		video_dict[video_id] = video_title
	print("added " + str(len(videos)) + " videos to a total of " + str(len(video_dict.keys())))
	return token

KEYWORD = "slack"
DATA_DIR = "data"

token = grab_videos(KEYWORD)
for id, title in video_dict.items():
	print(id, " - " ,title)

def srt_time_to_ms(h, m, s, ms):
	converted = int(ms)
	converted += (1000 * int(s))
	converted += (1000 * 60 * int(m))
	converted += (1000 * 60 * 60 * int(h))
	return converted

url_template = "http://youtube.com/watch?v={}"
ffmpeg_template = "ffmpeg -i {0}.mp4 -codec:a pcm_s16le -ac 1 {0}.wav"

tag_cleanr = re.compile('<.*?>|\(.*?\)|\[.*?\]')
srt_time_parser = re.compile("(\d+):(\d+):(\d+),(\d+)\s-->\s(\d+):(\d+):(\d+),(\d+)")
translator = str.maketrans('', '', string.punctuation)

for vid, title in video_dict.items():
	url = url_template.format(vid)
	print("\n", vid, title, url)
	yt = YouTube(url)
	caption = yt.captions.get_by_language_code('en')
	if caption:
		# retrieve audio file
		yt.streams.first().download(filename=vid)
		cmd = ffmpeg_template.format(vid).split()
		subprocess.check_output(cmd)
		audio = librosa.core.load(vid+".wav", 16000)[0]

		os.remove(vid+".mp4")
		os.remove(vid+".wav")

		dir_name = DATA_DIR+"/"+vid
		os.makedirs(dir_name)

		cc_arr = caption.generate_srt_captions().split('\n\n')
		for cc in cc_arr:
			cc_index, cc_time, cc_text = cc.split('\n')
			cc_text = tag_cleanr.sub('', cc_text).strip()

			if KEYWORD not in cc_text:
				continue

			cc_text = cc_text.translate(translator) # clean up punctuation

			match_result = srt_time_parser.match(cc_time)
			if match_result:
				start_time_ms = srt_time_to_ms(match_result.group(1), match_result.group(2), match_result.group(3), match_result.group(4))
				stop_time_ms = srt_time_to_ms(match_result.group(5), match_result.group(6), match_result.group(7), match_result.group(8))
				
				start_pos = start_time_ms * 16
				stop_pos = stop_time_ms * 16

				block = audio[start_pos:stop_pos] # *16 since 16 samples are captured per each ms

				# print(cc_time)
				# print(start_time_ms, stop_time_ms, stop_time_ms - start_time_ms)
				# print(start_pos, stop_pos, stop_pos - start_pos)
				# print(len(block))

				block_file_name = cc_text.replace(' ', '-')+".wav"
				librosa.output.write_wav(dir_name+"/"+block_file_name, block, 16000)
			else:
				print("parsing srt time failed - " + cc_time)
				sys.exit()

