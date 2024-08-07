# Introduction
This repo applies the PPO algorithm on the FrozenLake environment.

If the repo helps you, star it 🤗

# Install

## Envs

```bash
conda create -n ppo python=3.9
conda activate ppo
```

If  you have GPU, Please install pytorch corresponding to the CUDA version because it could faster your training.

If you only have CPU for training, That's OK

I have tried the  

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113		#for CUDA 11.4
```

and torch==2.1.0+cu118, both could work.

## Requirements

```bash
git clone https://github.com/Elapsedf/FrozenLake.git
pip install -r requirements.txt
```

**When you want to train a new model in Stochasitic Mode, Please make sure your GPU memory is Enough, If you find CUDA out of memroy, Try to decline the `update_freq`. As this parameter gets larger, it will consume more memory**



# File List

## Make_gif.py

This file contain the train and test mode, you only need to **change this file in the file end and run it**

## PPO.py

Main Algorithm

# Train

Please change the work dir to your own path

```python
save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std, pretrained)
save_gif(env_name)
list_gif_size(env_name)
```

**Note that  Remember to change the test_num when you want to train a new model**

# Test

```
test()		
save_gif(env_name)
list_gif_size(env_name)
```

**Note that  Remember to change the pre-trained model  and the gif_num when you want to test a new model**



# Pretrained Model

I trained 2 Model:**Determinstic** and **Stochastic**, both are in the `PreTrained` dir

They correspond to different settings of a parameter of the environment, As the following

```python
env = gym.make("FrozenLake-v1", is_slippery=False)	#Deterministic
env = gym.make("FrozenLake-v1")		# Stochastic
```

Deterministic Mode: The State Transition is depend on the Action which is chosen by the Actor Network

Stochastic Mode:State transfer in the environment is a stochastic process, i.e. the final outcome of the state transition does not depend only on the action but is also influenced by the environment

The following are the train result, and the ideal result is **1**



　　<table style="border:none;text-align:center;width:auto;margin: 0 auto;">
	<tbody>
		<tr>
			<td style="padding: 6px"><img src="http://cdn.elapsedf.cn/202311161011185.png" ></td>
		</tr>
        <tr><td><center><strong>Figure1: Deterministic Result</strong></center></td></tr>
	</tbody>
</table>


but in the Stochastic mode the result is not so good for the environment effect

So I Training in two phases

- In the First phases,I train a model and save it
- In the Second phases, I train a new model base on the previous model in first phases

So the Results are the following

　　<table style="border:none;text-align:center;width:auto;margin: 0 auto;">
	<tbody>
		<tr>
			<td style="padding: 6px"><img src="http://cdn.elapsedf.cn/202311161024572.png" ></td><td><img src="http://cdn.elapsedf.cn/202311161031224.png" ></td>
		</tr>
        <tr><td><center><strong>Figure2:First Phase</strong></center></td><td><center><strong>Figure3:Second Phase(Final Result)</strong></center></td></tr>
	</tbody>
</table>

# Result

**The gif result please see the gif dir**
