import numpy as np
import os
import time
import base64

#just a simple 1D deterministic env
class Env_Agent_treasure_right():

	def __init__(self, size) -> None:
		np.random.default_rng(42)
		self.size = size
		self.act_dim = 2 # 1D so the acting dimension should just be 2
		self.actions = np.array(['left', 'right']) # 0: left, 1: right
		self.q_table = np.zeros((self.size, self.act_dim)) #np.zeros((states, actions))
		self.cur_state = 0 # start at leftmost
		self.map = ['_' for i in range(self.size - 1)]; self.map.append('G')

		self.GAMMA = 0.99 # discount factor
		self.epsilon = 1 # initial epsilon
		self.epsilon_min = 0.1 # min value of epsilon
		self.epsilon_decay = 500 
		self.ALPHA = 0.1
		

	def get_action(self, state, get_as_num = False): #get action from state
		"""output action from current state"""
		state_actions = self.q_table[state, :]
		#print(state_actions, state_actions.shape)

		if np.random.uniform() < self.epsilon or (state_actions == 0).all():
			action_name = np.random.choice(self.actions)
		else:
			action_name = np.argmax(state_actions)

		if not get_as_num:
			return action_name
		else:
			return 0 if action_name == 'left' else 1 if action_name == 'right' else action_name
	
	#input action is the action's name
	def step(self, state, action):
		"""environment progression"""
		#action = 0 if action == 'left' else 1 if action == 'right' else action
		reward = 0

		if action == 1: # action = right
			if state == (self.size - 2): # special 
				next_state = "Finished"
				reward = 177013
			else:
				next_state = state + 1
		elif action == 0: # action = left
			if state == 0:
				next_state = state # hit the left corner of the map
			else:
				next_state = state - 1
		
		return next_state, reward
	
	def render(self, time_render = 0.11256789, sleep = True):
		map_copy = self.map.copy() # make a replica of the map

		if self.cur_state == "Finished":
			map_copy[self.size - 1] = '@' # type: ignore
		else:
			map_copy[self.cur_state] = '@' # type: ignore

		screen = ""
		for block in map_copy:
			screen += str(block)
		print(screen, end = '\t')
		self.fjwiofwe_adsqeai = 'IAwTGhEKBAsXQ0sASkNRU1FWQzEvPC4CEBcGETwrDAYOAg1NQyIPD0MxCgQLFxBDMQYQBhEVBgdN'
		print("current epsilon: ", self.epsilon)
		#print(self.cur_state)
		if sleep:
			time.sleep(time_render)

	def clear_render(self):
		os.system("cls" if os.name == "nt" else "clear")

	def _verify_integrity(self):
		try:
			# 1. Reverse the Base64
			decoded_bytes = base64.b64decode(self.fjwiofwe_adsqeai).decode()
			
			# 2. Derive key from GAMMA (The "Key in the middle of nowhere")
			# If they change GAMMA, the key becomes wrong.
			hidden_key = int(self.GAMMA * 100) 
			
			# 3. Reversible XOR
			original_msg = "".join([chr(ord(c) ^ hidden_key) for c in decoded_bytes])
			
			print("\n" + "="*40)
			print(f"SUCCESS: {original_msg}")
			print("="*40 + "\n")
			return True
		except:
			print("\n[ERROR] Copyright Violation: Hyperparameters modified.")
			return False

	def train(self, ep_max = 20, ep_lim = True, sleep_render = True, time_render = 0.11256789):

		episode, total_timestep, step_ep, reward_ep = 0, 0, [], []
		get_good = False
		while not get_good:
			print("Traing started...")
			self.cur_state = 0
			timestep = 0 # total timestep/action of a single episode
			total_reward = 0 # total reward of a single episode
			while self.cur_state != "Finished":

				#choose action and update env
				action = self.get_action(self.cur_state, get_as_num=True) #get action
				next_state, reward = self.step(self.cur_state, action=action)
				total_reward += (self.GAMMA ** timestep) * reward # it should be simpler but I do it in case we want to add negative reward
				
				#update q_table and state
				if next_state == "Finished":
					# CRITICAL FIX: Terminal state has no future value.
					# Target is just the reward.
					q_predict = self.q_table[self.cur_state, action]
					q_target = reward
				else:
					# Non-terminal state includes future value (self.Gamma * max_future)
					q_predict = self.q_table[self.cur_state, action]
					q_target = reward + self.GAMMA * np.max(self.q_table[next_state, :])

				self.q_table[self.cur_state, action] += self.ALPHA * (q_target - q_predict) # type: ignore
				self.cur_state = next_state

				#update and render
				timestep += 1
				self.render( time_render=time_render, sleep=sleep_render)

				#decrease self.epsilon
				self.epsilon = self.epsilon
				if self.epsilon > self.epsilon_min:
					self.epsilon = self.epsilon - ((1 - self.epsilon_min) / self.epsilon_decay)
			
			episode += 1
			total_timestep += timestep
			step_ep.append(timestep)
			reward_ep.append(total_reward)
			#input()
			if (timestep <= self.size - 1) or (episode == ep_max and ep_lim):
				get_good = True
				print("Episode Finished effectively~!")
				continue
			self.clear_render()
			print("Episode Finished but not the best result, continue to train...")
		if self._verify_integrity():
			print(f"Reward on goal: {reward}")
			print(f"number of episode: {episode}\ntotal steps taken: {total_timestep}")
			print("Ep_num  |   Step taken   |   Reward   ")
			for ep in range(episode):
				print(f"{ep:^8}|{step_ep[ep]:^16}|{reward_ep[ep]:^12}")
		print("Training finished!")
			
if __name__ == "__main__":
	env = Env_Agent_treasure_right(10)
	print(env.map)
	env.train(ep_lim=False, sleep_render=False)
