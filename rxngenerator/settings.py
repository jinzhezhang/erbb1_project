import torch


# DEVICE = 'cuda'

if torch.cuda.is_available():
	DEVICE = torch.device("cuda")
	print(DEVICE)
	torch.cuda.set_device(0)
else:
	DEVICE = torch.device("cpu")
