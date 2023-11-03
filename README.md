Hello and welcome to our 5.th semester project. This project is dedicated continue the development of maskable PPO solutions for the MicroRTS environment. 
There currently exist two solutions: Mrts_PPO_p5, which is by far the most effective solution and much simpler than the second solution which wraps the MicroRTS-environment in a gym-environment in order to access the SB3-contrib implementation of maskedPPO. The SB3 solution requires strictly dependent on several deprecated modules and it is there highly recommended that you use the first soltuion in combination with Torch==2.0.1, as it is able to run environments in parallel, versus, SB3, which can only run a single. Mrts_PPO_p5 also has an SPS of 1400+, where the SB3-solution has an SPS of 13.

===========================================================SB3 solution guide===========================================================
Setup guide for SB3-solution:
1. Lav en my mappe med et nyt env:
2. python3.9 -m venv --without-pip myenv
3. source myenv/bin/activate
4. wget https://bootstrap.pypa.io/get-pip.py
5. python get-pip.py
6. MicroRTS install: https://github.com/Farama-Foundation/MicroRTS-Py
7. Gå et directory tilbage og Install sb3 contrib: git clone -b v1.3.0 https://github.com/Stable-Baselines-Team/stable-baselines3-contrib.git
8.Sæt den til at køre med 1.1.0 stable lines i setup.py
9. Kør med Masked_PPO og deal med problemerne!

Soft fork solution required to make gym run with MicroRTS:
Issue 2 løsningkode: 

ep_len = len(self.rewards)
ep_time = time.time() - self.t_start
ep_info = {"r": round(float(ep_rew), 6), "l": ep_len, "t": round(ep_time, 6)}

Issue 3: set training mode:
 -Comment den out når den popper up

===========================================================Mrts_PPO_p5 guide===========================================================
1. Install MicroRTS by following the official gude: https://github.com/Farama-Foundation/MicroRTS-Py
2. install Torch version 2.0.0 for better performance on GPU (optional).
3. Add the Mrts_PPO_p5 folder to your MicroRTS folder.
4. If not in the Virtual environment of, then enter it and execute main.Py to run a 8x8 map training with 24 parallel environments.
5. Files denote what you are able to modify to suit your needs, though it is only recommended to change the arguments of your session as well as tweak the environment to suit your needs.
6. Evaluate.py is used to evaluate models according to MicroRTS-league standards. Simply provide a path to the model you to evaluate and let the magic do its work. 
