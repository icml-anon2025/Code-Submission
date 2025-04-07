import torch
import pickle
import yaml
import numpy as np
import src.main_model_table_ft as m
import time  # Import time module

# Record experiment start time
start_time = time.time()
formatted_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))


# Load configuration
with open("./TabCSDI/config/online_ed_ft.yaml", "r") as f:
    config = yaml.safe_load(f)

# Define model path and dataset name
model_path = "./TabCSDI/save/online_ed_fold5/model.pth"
exe_name = "online_ed"

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load model
model = m.TabCSDI(exe_name, config, device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load generated samples from pickle file
sample_path = "./TabCSDI/save/online_ed_fold5/generated_outputs_nsample100.pk"
with open(sample_path, "rb") as f:
    tens = pickle.load(f)  # This should be a PyTorch tensor

print(f"Loaded Sample Shape: {tens[0].shape}")

# Initialize tokenizer and recovered list
tokenizer = model.tokenizer
recovered_list = []

# Recover samples
for i in range(tens[0].shape[0]):
    # print(i)
    recovered_sample = tokenizer.recover(tens[0][i], d_numerical=3)
    recovered_list.append(recovered_sample.unsqueeze(0))

# Combine all recovered samples
recovered_samples = torch.cat(recovered_list, dim=0)
print(f"Recovered Shape: {recovered_samples.shape}")

end_time = time.time()

# Save recovered samples as a pickle file
save_path = f"./TabCSDI/save/online_ed_fold5/detoken_samples_output_online_ed_2.pk"
with open(save_path, "wb") as f:
    pickle.dump(recovered_samples.cpu(), f)  # Ensure it's saved in CPU memory

# print(f"Recovered samples saved to {save_path}")
# print(f"Final Recovered Shape: {recovered_samples.shape}")

# Record experiment end time


# Compute total execution time
# execution_time = end_time - start_time

# Print experiment duration
# print(f"Experiment started at: {time.ctime(start_time)}")
# print(f"Experiment ended at: {time.ctime(end_time)}")
# print(f"Total execution time: {execution_time:.2f} seconds")