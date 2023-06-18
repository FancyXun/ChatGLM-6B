from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
from torchviz import make_dot
from torch import tensor

"""
介绍下贵州

{'input_ids': tensor([[     5,  64474,  63880,  68550, 130001, 130004]]), 'past_key_values': None, 'position_ids': tensor([[[0, 1, 2, 3, 4, 4],
         [0, 0, 0, 0, 0, 1]]]), 'attention_mask': tensor([[[[False, False, False, False, False,  True],
          [False, False, False, False, False,  True],
          [False, False, False, False, False,  True],
          [False, False, False, False, False,  True],
          [False, False, False, False, False,  True],
          [False, False, False, False, False, False]]]])}
"""
input_ids = tensor([[5, 64474, 63880, 68550, 130001, 130004]])
position_ids = tensor([[[0, 1, 2, 3, 4, 4], [0, 0, 0, 0, 0, 1]]])
attention_mask = tensor([[[[False, False, False, False, False, True],
                           [False, False, False, False, False, True],
                           [False, False, False, False, False, True],
                           [False, False, False, False, False, True],
                           [False, False, False, False, False, True],
                           [False, False, False, False, False, False]]]])
model_inputs = {
    'input_ids': input_ids, 'past_key_values': None, 'position_ids': position_ids, attention_mask: attention_mask
}

tokenizer = AutoTokenizer.from_pretrained("./chatglm", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm", trust_remote_code=True).float()
model = model.eval()

print(model)

y = model(model_inputs['input_ids'],
          return_dict=True,
          output_attentions=False,
          output_hidden_states=False, )

# for k, v in dict(list(model.named_parameters())).items():
#     print(type(k), type(v))

k_v = [(i[0], i[1].data) for i in list(model.named_parameters())]

modelVis = make_dot(y.logits, params=dict(k_v + [('x', model_inputs['input_ids'])]))
modelVis.format = "svg"
modelVis.directory = "data"
modelVis.view()
