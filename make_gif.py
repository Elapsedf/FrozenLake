import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
from PIL import Image

import gym
#import roboschool

# import pybullet_envs
import pdb
from PPO import PPO
import matplotlib.pyplot as plt
import pandas as pd
import shutil
os.environ["SDL_VIDEODRIVER"] = "dummy"



"""
One frame corresponding to each timestep is saved in a folder :

PPO_gif_images/env_name/000001.jpg
PPO_gif_images/env_name/000002.jpg
PPO_gif_images/env_name/000003.jpg
...
...
...


if this section is run multiple times or for multiple episodes for the same env_name;
then the saved images will be overwritten.

"""

############################# save images for gif ##############################

dir ="/mnt/data2/lihaoyuan/zdf/UAV-control/"

def plot_result(csv_name,flag):
    # result_path="/home/jack/Project/zdf/DIMST/PPO/results/"
    # csv_name="secure_rate_irs80_user5_snr1_penalty3_channel_sigma2_log_12272138_irs[40,-100,0].csv"
    # fig_name="secure_rate_irs80_user5_snr1_penalty3_channel_sigma2_log_12272138_irs[40,-100,0].png"
    #result_reward = result_path+csv_name # 替换为实际的文件路径
    file_root, file_extension = os.path.splitext(csv_name)
    
    # 检查文件是否以.csv结尾
    if file_extension.lower() == '.csv':
        # 构建新的文件名，将.csv替换为新的扩展名
        fig_name = file_root + '.png'
        # 重命名文件
        # os.rename(file_path, new_file_path)
        # print(f'文件已重命名为: {new_file_path}')
    else:
        print('文件不是以.csv结尾，无法更改扩展名。')
        
    df = pd.read_csv(csv_name)

    # 提取数据
    episode = df['episode']
    time_step = df['time_step']
    mean_return = df['mean_return']
    
    # 绘制曲线
    plt.figure(figsize=(10, 6))
    #plt.plot(episode, return_value, label='Return')
    plt.plot(episode, mean_return, label='Mean Return')

    # 添加标签和标题
    plt.xlabel('Episode')
    if flag==1:
        plt.ylabel('Rate')
        plt.title('Rate and Mean Rate vs. Episode')
    elif flag==0:
        plt.ylabel('Return')
        plt.title('Mean Return vs. Episode')
    plt.legend()  # 显示图例
    
    # 显示图形
    plt.savefig(fig_name)
	
	#plt.close()  # 就是这里 一定要关闭
    plt.close()
    #plt.pyplot.close()


def save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std, pretrained):

	print("============================================================================================")

	


	#K_epochs = 80               # update policy for K epochs
	K_epochs = 80
	eps_clip = 0.2              # clip parameter for PPO
	gamma = 0.99                # discount factor

	lr_actor = 0.0003         # learning rate for actor
	lr_critic = 0.001         # learning rate for critic

	
	env = gym.make(env_name)

	# state space dimension
	#state_dim = env.observation_space.n
	state_dim=1
	# action space dimension
	if has_continuous_action_space:
		action_dim = env.action_space.shape
	else:
		action_dim = env.action_space.n


	
	# make directory for saving gif images
	gif_images_dir = dir+"PPO_gif_images" + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make environment directory for saving gif images
	gif_images_dir = gif_images_dir + '/' + env_name + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make directory for gif
	gif_dir = dir+"PPO_gifs" + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	# make environment directory for gif
	gif_dir = gif_dir + '/' + env_name  + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)



	ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

	reward_path=dir + "PPO_results/"
	if not os.path.exists(reward_path):
		os.mkdir(reward_path)
	
	reward_path = reward_path + f"test{test_num}_episode{total_test_episodes}"+ ".csv"

	res_r = open(reward_path, "w+")
	res_r.write('episode,time_step,mean_return\n')
	# preTrained weights directory

	


	directory = dir+"PPO_preTrained" + '/' 
	if not os.path.exists(directory):
		os.mkdir(directory)
	directory=directory+ env_name + '/'
	if not os.path.exists(directory):
		os.mkdir(directory)
	checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
	#print("loading network from : " + checkpoint_path)
	if pretrained:
		ppo_agent.load(checkpoint_path)
	checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, save_num_pretrained)
	print("--------------------------------------------------------------------------------------------")

	img_freq = 100
	
	test_running_reward = 0
	last_avg_reward = 0
	last_reward  = 0
	for ep in range(1, total_test_episodes+1):

		ep_reward = 0
		state = env.reset()
		flag=0
		for t in range(1, max_ep_len+1):
			action = ppo_agent.select_action(state)
			#pdb.set_trace()
			state, reward, done, _ = env.step(action)
			ep_reward += reward
			
			ppo_agent.buffer.rewards.append(reward)
			ppo_agent.buffer.is_terminals.append(done)
        	

			if done ==True:
				flag=1				
			if ep ==total_test_episodes:
				img = env.render(mode = 'rgb_array')

				img = Image.fromarray(img)
				img_dir_path=gif_images_dir +  f"test{test_num}/"
				if not os.path.exists(img_dir_path):
					os.mkdir(img_dir_path)
				
				if flag==1:
					img_path=img_dir_path+f"episode{ep}_t_{str(t).zfill(6)}"  + '_done.jpg'
				else:
					img_path=img_dir_path +  f"episode{ep}_t_{str(t).zfill(6)}"  + '.jpg'
				
				img.save(img_path)
			if ep % log_freq ==0:
				avg_reward = test_running_reward / ep
				res_r.write('{},{},{}\n'.format(ep, t, avg_reward))                
				res_r.flush()

			if ep % plot_freq == 0:
				plot_result(reward_path,0)
			
			
			
			if done:
				break
		
		# clear buffer
		#ppo_agent.buffer.clear()
		test_running_reward +=  ep_reward
		if ep % save_model_freq == 0:
			val_reward = ep_reward / t	#即考虑当前episode是否以最快速度达到reward
			avg_reward2 = test_running_reward /ep
			if (val_reward>last_reward) or ((val_reward >= last_reward) and (avg_reward2 >= last_avg_reward)):	#存在概率事件，存储使mean更好的model
				print("current_episode_mean_reward : ", val_reward)
				print("Test average reward: ", avg_reward2)
				print(
					"--------------------------------------------------------------------------------------------")
				print("saving model at : " + checkpoint_path)
				ppo_agent.save(checkpoint_path)
				print("model saved")
				print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
				print(
					"--------------------------------------------------------------------------------------------")

				last_reward = val_reward
				last_avg_reward = avg_reward2

		if ep % update_freq ==0:
			ppo_agent.update()
		
		print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
		ep_reward = 0
		torch.cuda.empty_cache()



	env.close()



	print("============================================================================================")

	print("total number of frames / timesteps / images saved : ", t)

	avg_test_reward = test_running_reward / total_test_episodes
	avg_test_reward = round(avg_test_reward, 2)
	print("average test reward : " + str(avg_test_reward))

	print("============================================================================================")







######################## generate gif from saved images ########################

def save_gif(env_name):

	print("============================================================================================")

	

	# adjust following parameters to get desired duration, size (bytes) and smoothness of gif
	total_timesteps = int(300)	#300
	step = 1
	frame_duration = 150


	# input images
	gif_images_dir = dir + "PPO_gif_images/" + env_name + f"/test{test_num}" +'/*.jpg'


	# ouput gif path
	gif_dir = dir +"PPO_gifs" 
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + env_name
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_path = gif_dir  + f"/test{test_num}_gif_{gif_num}" + '.gif'



	img_paths = sorted(glob.glob(gif_images_dir))
	#img_paths = img_paths[-total_timesteps:]		#[0-299],取前300张
	img_paths = img_paths[:total_timesteps]
	img_paths = img_paths[::step]


	print("total frames in gif : ", len(img_paths))
	print("total duration of gif : " + str(round(len(img_paths) * frame_duration / 1000, 2)) + " seconds")



	# save gif
	img, *imgs = [Image.open(f) for f in img_paths]
	img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration, loop=0)

	print("saved gif at : ", gif_path)



	print("============================================================================================")






############################# check gif byte size ##############################

def list_gif_size(env_name):

	print("============================================================================================")

	gif_dir = dir + "PPO_gifs/" + env_name + '/*.gif'

	gif_paths = sorted(glob.glob(gif_dir))

	for gif_path in gif_paths:
		file_size = os.path.getsize(gif_path)
		print(gif_path + '\t\t' + str(round(file_size / (1024 * 1024), 2)) + " MB")


	print("============================================================================================")


def test(env_name, try_time, reach_goal_time):
	K_epochs = 80               # update policy for K epochs
	eps_clip = 0.2              # clip parameter for PPO
	gamma = 0.99                # discount factor

	lr_actor = 0.0003         # learning rate for actor
	lr_critic = 0.001         # learning rate for critic
	env = gym.make(env_name,is_slippery=False)
	if has_continuous_action_space:
		action_dim = env.action_space.shape
	else:
		action_dim = env.action_space.n
	
	state_dim=1
	directory = dir+"PPO_preTrained" + '/' + env_name + '/'
	checkpoint_path = directory + "PPO_{}_{}_{}_deterministic.pth".format(env_name, random_seed, run_num_pretrained)
	print("============================================================================================")
	print("Load the Pretrained Model: PPO_{}_{}_{}_deterministic.pth".format(env_name, random_seed, run_num_pretrained))
	print("============================================================================================")
	# make directory for saving gif images
	gif_images_dir = dir+"PPO_gif_images" + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make environment directory for saving gif images
	gif_images_dir = gif_images_dir + '/' + env_name + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make directory for gif
	gif_dir = dir+"PPO_gifs" + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	# make environment directory for gif
	gif_dir = gif_dir + '/' + env_name  + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)


	
	ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
	ppo_agent.load(checkpoint_path)
	ep_reward = 0
	state = env.reset()
	#flag=0
	for t in range(1, max_ep_len+1):
		img = env.render(mode = 'rgb_array')
		img = Image.fromarray(img)
		img_dir_path=gif_images_dir +  f"test{test_num}/"

		if not os.path.exists(img_dir_path):
			os.mkdir(img_dir_path)
		
		img_path=img_dir_path +  f"test_t_{str(t).zfill(6)}"  + '.jpg'
		img.save(img_path)
		action = ppo_agent.select_action(state)
		state, reward, done, _ = env.step(action)
		ep_reward += reward
		
		ppo_agent.buffer.rewards.append(reward)
		ppo_agent.buffer.is_terminals.append(done)
		# if done == True:
		# 	flag=1
		
		
		# if flag==1:
		# 	img_path=img_dir_path+f"test_t_{str(t).zfill(6)}"  + '_done.jpg'
		# else:
		# 	img_path=img_dir_path +  f"test_t_{str(t).zfill(6)}"  + '.jpg'
		
		if (done == True) or (t==max_ep_len):
			print("Reward :", ep_reward)
			try_time += 1
			if (ep_reward !=1) or t>=7:
				
				if ep_reward ==1:
					reach_goal_time += 1
				try:
        # 删除文件夹中的所有内容，包括子文件夹和文件
					shutil.rmtree(img_dir_path)
					print(f"Contents of {img_dir_path} deleted successfully.")
				except Exception as e:
					print(f"An error occurred: {e}")
				test(env_name, try_time, reach_goal_time)
			else:
				reach_goal_time += 1
				img = env.render(mode = 'rgb_array')
				img = Image.fromarray(img)
				img_path=img_dir_path +  f"test_t_{str(t+1).zfill(6)}"  + '.jpg'
				img.save(img_path)
				print("Reach the Goal Successfully!")
				print("Total try {} times".format(try_time))
				print("Total reach goal {} times".format(reach_goal_time))
			break
	#save_gif(env_name)
	




if __name__ == '__main__':


	# env_name = "CartPole-v1"
	# has_continuous_action_space = False
	# max_ep_len = 400
	# action_std = None


	# env_name = "LunarLander-v2"
	# has_continuous_action_space = False
	# max_ep_len = 500
	# action_std = None


	# env_name = "BipedalWalker-v2"
	# has_continuous_action_space = True
	# max_ep_len = 1500           # max timesteps in one episode
	# action_std = 0.1            # set same std for action distribution which was used while saving


	# env_name = "RoboschoolWalker2d-v1"
	# has_continuous_action_space = True
	# max_ep_len = 1000           # max timesteps in one episode
	# action_std = 0.1            # set same std for action distribution which was used while saving
	

	

	env_name = "FrozenLake-v1"
	has_continuous_action_space = False
	max_ep_len = 50          # max timesteps in one episode	
	total_test_episodes = int(10e6)    # save gif for only one episode	

	test_num=3					# set this to test different kinds of configs
	gif_num = 1     #### change this to prevent overwriting gifs in same env_name and test folder

	action_std = None           # set same std for action distribution which was used while saving

	update_freq = int(500)			# set this to change the freq of update agent	Note that this value shouldn't be too high, or CUDA will out of memroy 
	log_freq = int(1e3)				# set this to change the freq of lof reward
	plot_freq = int(1e3)				# set this to change the freq of plot result
	save_model_freq = int(1e3)		# set this to change the freq of save model

	random_seed = 0             #### set this to load a particular checkpoint trained on random seed
	run_num_pretrained = 0      #### set this to load a particular checkpoint num
	save_num_pretrained = 0		#### set this to save a particular checkpoint num
	start_time = datetime.now().replace(microsecond=0)

	pretrained = True			## set this to decide whether load a pretrained model
	#So the best mean reward = 1/6 = 0.166

	# env_name = "RoboschoolHopper-v1"
	# has_continuous_action_space = True
	# max_ep_len = 1000           # max timesteps in one episode
	# action_std = 0.1            # set same std for action distribution which was used while saving

	# save .jpg images in PPO_gif_images folder
	#save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std, pretrained)
	try_time = 0
	reach_goal_time = 0
	test(env_name, try_time, reach_goal_time)
	# save .gif in PPO_gifs folder using .jpg images
	save_gif(env_name)

	# list byte size (in MB) of gifs in one "PPO_gif/env_name/" folder
	list_gif_size(env_name)
