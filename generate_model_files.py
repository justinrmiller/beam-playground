from transformers import BertForMaskedLM
import torch

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
torch.save(model.state_dict(), 'model_a.pth')
torch.save(model.state_dict(), 'model_b.pth')
torch.save(model.state_dict(), 'model_c.pth')
torch.save(model.state_dict(), 'model_d.pth')
torch.save(model.state_dict(), 'model_e.pth')
