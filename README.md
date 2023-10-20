1. Lav en my mappe med et nyt env: 
		○ python3.9 -m venv --without-pip myenv
		○ source myenv/bin/activate
		○ wget https://bootstrap.pypa.io/get-pip.py
		python get-pip.py
2. MicroRTS install: https://github.com/Farama-Foundation/MicroRTS-Py
3. Gå et directory tilbage og Install sb3 contrib: git clone -b v1.3.0 https://github.com/Stable-Baselines-Team/stable-baselines3-contrib.git
		○ Sæt den til at køre med 1.1.0 stable lines i setup.py
4. Kør med Masked_PPO og deal med problemerne!


Problemer


Issue 2 løsning: 
	ep_len = len(self.rewards)
	ep_time = time.time() - self.t_start
	ep_info = {"r": round(float(ep_rew), 6), "l": ep_len, "t": round(ep_time, 6)}

Issue 3: set training mode:
	Comment den out når den popper up

